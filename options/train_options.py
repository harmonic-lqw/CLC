from argparse import ArgumentParser

from contrastive import util as util

class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # general setup
        self.parser.add_argument('--exp_dir', type=str,
                                 help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='BackboneEncoder', type=str,
                                 help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=6, type=int,
                                 help='Number of input image channels to the ReStyle encoder. Should be set to 6.')
        self.parser.add_argument('--output_size', default=1024, type=int,
                                 help='Output size of generator')

        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int,
                                 help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float,
                                 help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str,
                                 help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool,
                                 help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--lpips_lambda', default=0, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0, type=float,
                                 help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float,
                                 help='W-norm loss multiplier factor')
        self.parser.add_argument('--moco_lambda', default=0, type=float,
                                 help='Moco feature loss multiplier factor')

        self.parser.add_argument('--stylegan_weights', default=None, type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to ReStyle model checkpoint')

        self.parser.add_argument('--max_steps', default=500000, type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int,
                                 help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int,
                                 help='Model checkpoint interval')

        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,
                                 help='Number of forward passes per batch during training')
        self.parser.add_argument('--use_con', action='store_true', help='Whether to use contrastive')
        self.parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        self.parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        self.parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        self.parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        self.parser.add_argument('--lambda_MS_neg', type=float, default=1.0, help='weight for NEG DIVERSITY loss')
        self.parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        self.parser.add_argument('--netF_nc', type=int, default=256)
        self.parser.add_argument('--con_lambda', default=0.5, type=float, help='Contrastive loss multiplier factor')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
