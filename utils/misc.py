import os 
import bezier
import subprocess
import numpy as np

def recog_indices_to_str(recog_indices, chars):
    recog_str = []
    for idx in recog_indices:
        if idx < len(chars):
            recog_str.append(chars[idx])
        else:
            break 
    return ''.join(recog_str)

def sample_bezier_curve(bezier_pts, num_points=10, mid_point=False):
    curve = bezier.Curve.from_nodes(bezier_pts.transpose())
    if mid_point:
        x_vals = np.array([0.5])
    else:
        x_vals = np.linspace(0, 1, num_points)
    points = curve.evaluate_multi(x_vals).transpose()
    return points 

def bezier2bbox(bezier_pts):
    bezier_pts = bezier_pts.reshape(8, 2)
    points1 = sample_bezier_curve(bezier_pts[:4], 20)
    points2 = sample_bezier_curve(bezier_pts[4:], 20)
    points = np.concatenate((points1, points2))
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    return [xmin, ymin, xmax, ymax]

def decode_seq(seq, type, args):
    seq = seq[seq != args.padding_index]
    if type == 'input':
        seq = seq[1:]
    elif type == 'output':
        seq = seq[:-1]
    elif type == 'none':
        seq = seq 
    else:
        raise ValueError
    seq = seq.reshape(-1, 27)

    decode_result = []
    for text_ins_seq in seq:
        point_x = text_ins_seq[0] / args.num_bins
        point_y = text_ins_seq[1] / args.num_bins
        recog = []
        for index in text_ins_seq[2:]:
            if index == args.recog_pad_index:
                break 
            if index == args.recog_pad_index - 1:
                continue
            recog.append(args.chars[index - args.num_bins])
        recog = ''.join(recog)
        decode_result.append(
            {'point': (point_x.item(), point_y.item()), 'recog': recog}
        )
    return decode_result

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message