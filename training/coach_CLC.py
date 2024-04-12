import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger

from tqdm import tqdm

from contrastive.LearnedPatchNCELoss import LearnedPatchNCELoss
from contrastive.feature_extractor import define_F
from contrastive.negative_generator import define_N

def accumulate(model1, model2, decay=0.9):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0

		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		self.net = pSp(self.opts).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# get the image corresponding to the latent average
		self.avg_image = self.net(self.net.latent_avg.unsqueeze(0),
								  input_code=True,
								  randomize_noise=False,
								  return_latents=False,
								  average_code=True)[0]
		self.avg_image = self.avg_image.to(self.device).float().detach()
		if self.opts.dataset_type == "cars_encode":
			self.avg_image = self.avg_image[:, 32:224, :]
		common.tensor2im(self.avg_image).save(os.path.join(self.opts.exp_dir, 'avg_image.jpg'))

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		#########________CON________##########
		if self.opts.use_con:
			self.nce_layers = [0, 3, 12, 15] # [0, 3, 7, 13]:128-128、128-64、64-32、32-16
			# face
			# self.nce_layers = [0, 1, 3, 20, 23] # 0/(N, 64, 128, 128) (256-128)  3/(N, 128, 64, 64)  7/(N, 256, 32, 32)  20/(N, 256, 32, 32)  21/(N, 512, 16, 16)
			self.criterionNCE = []
			for i in range(len(self.nce_layers) + 2):
				self.criterionNCE.append(LearnedPatchNCELoss(opts).to(self.device))

			# (3, mlp_sample, instance, False, xavier, 0.02, 使用2stride卷积, 0, opt)
			self.gpu_ids = [0]
			self.netF = define_F(3, 'mlp_sample', 'instance', not self.opts.no_dropout, self.opts.init_type, self.opts.init_gain, self.opts.no_antialias, self.gpu_ids, self.opts)
			self.netN = define_N(self.nce_layers, 'neg_gen_momentum', self.opts.init_type, self.opts.init_gain, self.gpu_ids, self.opts)

			self.netF_ = define_F(3, 'mlp_sample', 'instance', not self.opts.no_dropout, self.opts.init_type, self.opts.init_gain, self.opts.no_antialias, self.gpu_ids, self.opts)
			self.netF_.train(False)

			self.optimizer_F, self.optimizer_N = self.configure_FandN_optimizers()

	#######################————NEGCUT————#######################

	def configure_FandN_optimizers(self):
		y_hat, latent = None, None
		for batch_idx, batch in enumerate(self.train_dataloader):
			x, y = batch
			x, y = x.to(self.device).float(), y.to(self.device).float()
			x_input = torch.cat([x, x], dim=1)
			y_hat, latent = self.net.forward(x_input, return_latents=True)
			if self.opts.dataset_type == "cars_encode":
				y_hat = y_hat[:, :, 32:224, :]
			self.compute_F_loss(x, y_hat)
			break
		optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opts.lr, betas=(self.opts.beta1, self.opts.beta2))
		optimizer_N = torch.optim.Adam(self.netN.parameters(), lr=self.opts.lr, betas=(self.opts.beta1, self.opts.beta2))
		return optimizer_F, optimizer_N

	def compute_N_loss(self, x, y_hat):
		# 计算NCE
		# y_hat_clone = y_hat.clone().detach().requires_grad_(True)
		src_input = torch.cat([x, x], dim=1)
		tgt_input = torch.cat([y_hat, y_hat], dim=1)
		
		self.loss_NCE = self.calculate_NCE_loss(src_input, tgt_input, use_neg=True)

		# 计算Ldiv
		total_loss = 0.0
		n_layers = len(self.criterionNCE) + 2
		for n_k in self.neg_k_pool:
			num_patches = self.opts.num_patches
			# (N, 256, 256)
			n_k = n_k.view(-1, num_patches, n_k.shape[1])
			loss = - torch.abs(n_k[:, :num_patches // 2] - n_k[:, num_patches // 2:]).mean()
			total_loss += loss.mean() * self.opts.lambda_MS_neg
		loss_MS_noise = total_loss / n_layers
		
		self.loss_N = - self.loss_NCE + loss_MS_noise

		return self.loss_N
	
	def compute_F_loss(self, x, y_hat):
		# y_hat_clone = y_hat.clone().detach().requires_grad_(True)
		src_input = torch.cat([x, x], dim=1)
		tgt_input = torch.cat([y_hat, y_hat], dim=1)
		self.loss_NCE = self.calculate_NCE_loss(src_input, tgt_input)
		return self.loss_NCE

	def calculate_NCE_loss(self, src, tgt, use_neg=False):
        # use_neg:训练G和F为False、训练N为True

		n_layers = len(self.nce_layers) + 2

        # feat_q/k是输入经过生成器对应层之后得特征图大小: [6, N, C, H, W]
        # (N, 3, 256, 256)，(N, 64, 256, 256)，(N, 64, 128, 128)，(N, 128, 64, 64)，(N, 128, 64, 64), (N, 256, 32, 32), (N, 256, 32, 32)
		feat_q = self.net.encoder(tgt, self.nce_layers, encode_only=True)

		feat_k = self.net.encoder(src, self.nce_layers, encode_only=True)

		# sample_ids 记录位置
		# [6, N*256, 256]
		feat_k_pool, sample_ids = self.netF(feat_k, self.opts.num_patches, None)
        
        # 中间256表示采样的256个像素点，第二个256表示q、k、neg的维度
        # feat_q/k_pool 经过H后的特征向量[6, N*256, 256]
		feat_q_pool, _ = self.netF(feat_q, self.opts.num_patches, sample_ids)
        # 此时找到了每个层、每个样本、256个采样的像素、对应的长度为256的q和k

		# [6, N, 256, H, W]
		neg_k_pool, _ = self.netF_(feat_k, num_patches=0)
		# [6, N*256, 256]
		self.neg_k_pool = self.netN(neg_k_pool, self.opts.num_patches)

		total_nce_loss = 0.0
		for nce_id, (f_q, f_k, n_k, crit, nce_layer) in enumerate(zip(feat_q_pool, feat_k_pool, self.neg_k_pool, self.criterionNCE, self.nce_layers + [-1])):
			if use_neg:
				loss = crit(f_q.detach(), f_k.detach(), n_k) * self.opts.lambda_NCE
			else:
				loss = crit(f_q, f_k, n_k.detach()) * self.opts.lambda_NCE
			total_nce_loss += loss.mean()
		
		return total_nce_loss / n_layers

	def set_requires_grad(self, nets, requies_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requies_grad

	#######################————NEGCUT————#######################

	def perform_train_iteration_on_batch(self, x, y):
		y_hat, latent = None, None
		loss_dict, id_logs = None, None
		y_hats = {idx: [] for idx in range(x.shape[0])}
		for iter in range(self.opts.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=None, return_latents=True)
			else:
				y_hat_clone = y_hat.clone().detach().requires_grad_(True)
				latent_clone = latent.clone().detach().requires_grad_(True)
				x_input = torch.cat([x, y_hat_clone], dim=1)
				y_hat, latent = self.net.forward(x_input, latent=latent_clone, return_latents=True)

			# print('='*30)
			# print("y_hat1:", y_hat.size()) (8, 3, 256, 256)
			if self.opts.dataset_type == "cars_encode":
				y_hat = y_hat[:, :, 32:224, :]
			
			# print("y_hat2:", y_hat.size()) (8, 3, 192, 256)
			# print('y:', y.size()) (8, 3, 192, 256)
			# print('='*30)

			loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)

			if not self.opts.use_con:
				loss.backward()
				if iter == (self.opts.n_iters_per_batch - 1):
					self.optimizer.step()


			# con
			else:
				if iter <= 3:
					loss.backward()
					if iter == 3:
						self.optimizer.step()  # 如果这里不写step()而是在else后写，显存占用多4000左右
				else:
					# update N
					# print('='*30)
					# for param in self.net.parameters():
					# 	print('grad1: ', param.grad)
					# 	break
					self.set_requires_grad(self.netN, True)
					self.set_requires_grad(self.netF, False)
					self.set_requires_grad(self.net, False)
					self.optimizer_N.zero_grad()
					loss_N = self.compute_N_loss(x, y_hat)
					loss_N.backward()
					# if iter == (self.opts.n_iters_per_batch - 1):
					self.optimizer_N.step()

					# update F and pSp
					self.set_requires_grad(self.netN, False)
					self.set_requires_grad(self.netF, True)
					self.set_requires_grad(self.net, True)
					self.optimizer_F.zero_grad()
					self.optimizer.zero_grad()
					loss_F = self.compute_F_loss(x, y_hat)
					loss_encoder = loss + loss_F * self.opts.con_lambda
					# loss_encoder = loss_F * self.opts.con_lambda
					loss_encoder.backward()
					# if iter == (self.opts.n_iters_per_batch - 1):
					self.optimizer_F.step()
					self.optimizer.step()
				# if True:
					accumulate(self.netF_, self.netF)
				
					loss_dict['loss_N'] = float(loss_N)
					loss_dict['loss_F'] = float(loss_F)
					loss_dict['loss_encoder'] = float(loss_encoder)

			for idx in range(x.shape[0]):
				y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

		return y_hats, loss_dict, id_logs

	def train(self):
		self.net.train()
		if self.opts.use_con:
			self.netF.train()
			self.netN.train()
		par = tqdm(total=20000)
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				# if not self.opts.use_con:
				self.optimizer.zero_grad()

				x, y = batch
				x, y = x.to(self.device).float(), y.to(self.device).float()

				y_hats, loss_dict, id_logs = self.perform_train_iteration_on_batch(x, y)

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y, y_hats, title='images/train')

				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					# print('*'*30)
					# print('val_loss_dict.keys', val_loss_dict.keys())
					# print('*'*30)
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss_lpips'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss_lpips']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					par.update(1)
					par.close()
					break
					
				if self.global_step % 10 == 0:
					par.update(1)
				
				self.global_step += 1

				

	def perform_val_iteration_on_batch(self, x, y):
		y_hat, latent = None, None
		cur_loss_dict, id_logs = None, None
		y_hats = {idx: [] for idx in range(x.shape[0])}
		for iter in range(self.opts.n_iters_per_batch):
			if iter == 0:
				avg_image_for_batch = self.avg_image.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
				x_input = torch.cat([x, avg_image_for_batch], dim=1)
			else:
				x_input = torch.cat([x, y_hat], dim=1)

			y_hat, latent = self.net.forward(x_input, latent=latent, return_latents=True)

			if self.opts.dataset_type == "cars_encode":
				y_hat = y_hat[:, :, 32:224, :]

			loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)

			if self.opts.use_con:
				if iter == 4:
					loss_N = self.compute_N_loss(x, y_hat)

					loss_F = self.compute_F_loss(x, y_hat)
					loss_encoder = loss + loss_F * self.opts.con_lambda
					# loss_encoder = loss_F * self.opts.con_lambda

					cur_loss_dict['loss_N'] = float(loss_N)
					cur_loss_dict['loss_F'] = float(loss_F)
					cur_loss_dict['loss_encoder'] = float(loss_encoder)

			# store intermediate outputs
			for idx in range(x.shape[0]):
				y_hats[idx].append([y_hat[idx], id_logs[idx]['diff_target']])

		return y_hats, cur_loss_dict, id_logs

	def validate(self):
		self.net.eval()
		if self.opts.use_con:
			self.netF.eval()
			self.netN.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y = batch
			with torch.no_grad():
				x, y = x.to(self.device).float(), y.to(self.device).float()
				y_hats, cur_loss_dict, id_logs = self.perform_val_iteration_on_batch(x, y)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hats, title='images/test', subscript='{:04d}'.format(batch_idx))

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				if self.opts.use_con:
					self.netF.train()
					self.netN.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		if self.opts.use_con:
			self.netF.train()
			self.netN.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			raise Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
		print('Loading dataset for {}'.format(self.opts.dataset_type))
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, x, y, y_hat, latent):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			if type(y_hat) == dict:
				output_face = [
					[common.tensor2im(y_hat[i][iter_idx][0]), y_hat[i][iter_idx][1]]
					for iter_idx in range(len(y_hat[i]))
				]
			else:
				output_face = [common.tensor2im(y_hat[i])]
			cur_im_data = {
				'input_face': common.tensor2im(x[i]),
				'target_face': common.tensor2im(y[i]),
				'output_face': output_face,
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
		else:
			path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts),
			'latent_avg': self.net.latent_avg
		}
		return save_dict
