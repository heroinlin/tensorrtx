import os
import torch
import struct
import argparse
working_root = os.path.split(os.path.realpath(__file__))[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--model_pt',
                        type=str,
                        help='pytorch model file path',
                        default=os.path.join(working_root,
                                             "weights/yolov5l.pt"))
    parser.add_argument('-w',
                        '--model_wts',
                        type=str,
                        help='wts type model file path',
                        default=os.path.join(working_root,
                                             "weights/yolov5l.wts"))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    pt_model_path = args.model_pt
    wts_model_path = args.model_wts
    if not os.path.exists(pt_model_path):
        print("pytorch modle file is not exists! please check path!")
        exit()

    # Initialize
    device = torch.device('cpu')
    # Load model
    model = torch.load(pt_model_path, map_location=device)['model']  
    model.float().eval() # load to FP32
    model = model.to(device)

    f = open(wts_model_path, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')
    print("pt 2 wts success!")

if __name__ == '__main__':
    main()