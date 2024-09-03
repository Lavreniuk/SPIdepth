import sys
import io
import os
import torch
import torch.nn as nn
# import networks
#from options import MonodepthOptions
from collections import OrderedDict
from SQLdepth import SQLdepth, MonodepthOptions

def convert(opt, checkpoint_path, save_folder):
    model = SQLdepth(opt)
    print("loading checkpoint from {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] 

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0) 

    model.load_state_dict(torch.load(buffer, map_location='cpu'))

    buffer.close()

    encoder_state_dict = model.encoder.state_dict()
    encoder_state_dict['height'] = opt.height
    encoder_state_dict['width'] = opt.width
    encoder_state_dict['use_stereo'] = opt.use_stereo
    decoder_state_dict = model.depth_decoder.state_dict()
    os.makedirs(save_folder, exist_ok=True) # mkdir if not exits

    encoder_save_path = os.path.join(save_folder, "encoder.pth")
    decoder_save_path = os.path.join(save_folder, "depth.pth")
    torch.save(encoder_state_dict, encoder_save_path)
    torch.save(decoder_state_dict, decoder_save_path)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

if __name__ == '__main__':
    SQLdepth_options = MonodepthOptions()
    SQLdepth_options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        SQLdepth_opt_filename = '@' + sys.argv[1]
        opt = SQLdepth_options.parser.parse_args([SQLdepth_opt_filename])
    opt = SQLdepth_options.parser.parse_args([SQLdepth_opt_filename])
    opt.load_pretrained_model = False
    ckpt_path = "exps/inc_cityscapes_exps_v13/checkpoints/cvnXt_075_finetune_01-Aug_15-40-nodebs3-tep5-lr2e-06-wd0.01-569e0d45-bc5a-499a-9195-9f6f3d97fb01_best.pt"
    pth_path = "exps/inc_cityscapes_exps_v13"

    print("converting weights...")
    convert(opt, ckpt_path, pth_path)
    print("done.")


