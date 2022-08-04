import os 
import copy
import glob
import json
import numpy as np
import editdistance as ed

from tqdm import tqdm
from shapely.geometry import Point, LineString

def find_match_word(rec_str, lexicon, pair):
    rec_str = rec_str.upper()
    match_word = ''
    match_dist = 100
    for word in lexicon:
        word = word.upper()
        ed_dist = ed.eval(rec_str, word)
        norm_ed_dist = ed_dist / max(len(word), len(rec_str))
        if norm_ed_dist < match_dist:
            match_dist = norm_ed_dist
            if pair:
                match_word = pair[word]
            else:
                match_word = word
    return match_word, match_dist

def read_lexicon(lexicon_path):
    if lexicon_path.endswith('.txt'):
        lexicon = open(lexicon_path, 'r').read().splitlines()
        lexicon = [ele.strip() for ele in lexicon]
    else:
        lexicon = {}
        lexicon_dir = os.path.dirname(lexicon_path)
        num_file = len(os.listdir(lexicon_dir))
        assert(num_file % 2 == 0)
        for i in range(num_file // 2):
            lexicon_path_ = lexicon_path + f'{i+1:d}.txt'
            lexicon[i] = read_lexicon(lexicon_path_)
    return lexicon

def read_pair(pair_path):
    if 'ctw1500' in pair_path:
        return None

    if pair_path.endswith('.txt'):
        pair_lines = open(pair_path, 'r').read().splitlines()
        pair = {}
        for line in pair_lines:
            line = line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word)+1:]
            pair[word] = word_gt
    else:
        pair = {}
        pair_dir = os.path.dirname(pair_path)
        num_file = len(os.listdir(pair_dir))
        assert(num_file % 2 == 0)
        for i in range(num_file // 2):
            pair_path_ = pair_path + f'{i+1:d}.txt'
            pair[i] = read_pair(pair_path_)
    return pair    

def poly_center(poly_pts):
    poly_pts = np.array(poly_pts).reshape(-1, 2)
    num_points = poly_pts.shape[0]
    line1 = LineString(poly_pts[int(num_points/2):])
    line2 = LineString(poly_pts[:int(num_points/2)])
    mid_pt1 = np.array(line1.interpolate(0.5, normalized=True).coords[0])
    mid_pt2 = np.array(line2.interpolate(0.5, normalized=True).coords[0])
    return (mid_pt1 + mid_pt2) / 2

### official code
def include_in_dictionary(transcription):
    #special case 's at final
    if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
        transcription = transcription[0:len(transcription)-2]
    #hypens at init or final of the word
    transcription = transcription.strip('-');
    specialCharacters = str("'!?.:,*\"()·[]/");
    for character in specialCharacters:
        transcription = transcription.replace(character,' ')
    transcription = transcription.strip()
    if len(transcription) != len(transcription.replace(" ","")) :
        return False;
    if len(transcription) < 3:
        return False;
    notAllowed = str("×÷·");
    range1 = [ ord(u'a'), ord(u'z') ]
    range2 = [ ord(u'A'), ord(u'Z') ]
    range3 = [ ord(u'À'), ord(u'ƿ') ]
    range4 = [ ord(u'Ǆ'), ord(u'ɿ') ]
    range5 = [ ord(u'Ά'), ord(u'Ͽ') ]
    range6 = [ ord(u'-'), ord(u'-') ]
    for char in transcription :
        charCode = ord(char)
        if(notAllowed.find(char) != -1):
            return False
        valid = ( charCode>=range1[0] and charCode<=range1[1] ) or ( charCode>=range2[0] and charCode<=range2[1] ) or ( charCode>=range3[0] and charCode<=range3[1] ) or ( charCode>=range4[0] and charCode<=range4[1] ) or ( charCode>=range5[0] and charCode<=range5[1] ) or ( charCode>=range6[0] and charCode<=range6[1] )
        if valid == False:
            return False
    return True
    
def include_in_dictionary_transcription(transcription):
    #special case 's at final
    if transcription[len(transcription)-2:]=="'s" or transcription[len(transcription)-2:]=="'S":
        transcription = transcription[0:len(transcription)-2]
    #hypens at init or final of the word
    transcription = transcription.strip('-');            
    specialCharacters = str("'!?.:,*\"()·[]/");
    for character in specialCharacters:
        transcription = transcription.replace(character,' ')
    transcription = transcription.strip()
    return transcription

def read_gt(gt_folder, IS_WORDSPOTTING):
    gts = glob.glob(f"{gt_folder}/*.txt")
    gts.sort()

    gt_dict = {}
    for i in gts:
        lines = open(i, "r").readlines()
        imid = int(os.path.basename(i)[:-4])
        points = []
        recs = []
        dontcares = []
        for line in lines:
            if not line: 
                continue

            line_split = line.strip().split(",####")

            dontcare = False
            rec = line_split[1]
            if rec == "###":
                dontcare = True
            else:
                if IS_WORDSPOTTING:
                    if include_in_dictionary(rec) == False: 
                        dontcare = True
                    else:
                        rec = include_in_dictionary_transcription(rec)

            coords = line_split[0]
            coords = coords.split(",")
            coords = [int(ele) for ele in coords]
            center_pt = poly_center(coords)
            center_pt = Point(center_pt[0], center_pt[1])
            
            points.append(center_pt)
            recs.append(rec)
            dontcares.append(dontcare)
            matched = [0] * len(recs)

        gt_dict[imid] = [points, recs, matched, dontcares]

    return gt_dict

def read_result(result_path, lexicons, pairs, match_dist_thres, gt_folder, lexicon_type):
    results = json.load(open(result_path, 'r'))
    results.sort(reverse=True, key=lambda x: x['score'])

    results = [result for result in results if len(result['rec']) > 0]

    if not lexicons is None:
        print('Processing Results using Lexicon')
        new_results = []
        for result in tqdm(results):
            rec = result['rec']
            if lexicon_type == 2:
                lexicon = lexicons[result['image_id'] - 1]
                pair = pairs[result['image_id'] - 1]
            else:
                lexicon = lexicons
                pair = pairs

            match_word, match_dist = find_match_word(rec, lexicon, pair)
            if match_dist < match_dist_thres or \
               (('gt_ic13' in gt_folder or 'gt_ic15' in gt_folder) and lexicon_type == 0):
                rec = match_word
            else:
                continue
            result['rec'] = rec
            new_results.append(result)
        results = new_results    

    return results

def evaluate(results, gts, conf_thres):

    gts = copy.deepcopy(gts)
    results = copy.deepcopy(results)

    ngt = sum([len(ele[0]) for ele in gts.values()])
    ngt -= sum([sum(ele[3]) for ele in gts.values()])

    ndet = 0; ntp = 0
    for result in results:
        confidence = result["score"]
        if confidence < conf_thres:
            continue

        image_id = result['image_id']
        pred_coords = result["polys"]
        pred_rec = result["rec"]
        pred_point = Point(pred_coords[0][0], pred_coords[0][1])

        gt_imid = gts[image_id]
        gt_points = gt_imid[0]
        gt_recs = gt_imid[1]
        gt_matched = gt_imid[2]
        gt_dontcare = gt_imid[3]

        dists = [pred_point.distance(gt_point) for gt_point in gt_points]
        minvalue = min(dists)
        idxmin = dists.index(minvalue)
        if gt_recs[idxmin] == "###" or gt_dontcare[idxmin] == True:
            continue
        if pred_rec.upper() == gt_recs[idxmin].upper() and gt_matched[idxmin] == 0:
            gt_matched[idxmin] = 1
            ntp += 1

        ndet += 1

    if ndet == 0 or ntp == 0:
        recall = 0; precision = 0; hmean = 0
    else:
        recall = ntp / ngt
        precision  = ntp / ndet
        hmean = 2 * recall * precision / (recall + precision)
    return precision, recall, hmean, ntp, ngt, ndet

def main(args):

    if 'totaltext' in args.result_path.lower():
        gt_folder = 'evaluation/gt/gt_totaltext'; IS_WORDSPOTTING = True
        lexicon_paths = ['', 'evaluation/lexicons/totaltext/weak_voc_new.txt', ]
        pair_paths = ['', 'evaluation/lexicons/totaltext/weak_voc_pair_list.txt', ]
        lexicon_type = 1
    elif 'ctw1500' in args.result_path.lower():
        gt_folder = 'evaluation/gt/gt_ctw1500'; IS_WORDSPOTTING = False
        lexicon_paths = ['', 'evaluation/lexicons/ctw1500/weak_voc_new.txt', ]
        pair_paths = ['', 'evaluation/lexicons/ctw1500/weak_voc_pair_list.txt', ]
        lexicon_type = 1
    elif 'ic13' in args.result_path.lower():
        gt_folder = 'evaluation/gt/gt_ic13'; IS_WORDSPOTTING = False
        lexicon_paths = [
            'evaluation/lexicons/ic13/GenericVocabulary_new.txt',
            'evaluation/lexicons/ic13/ch2_test_vocabulary_new.txt',
            'evaluation/lexicons/ic13/new_strong_lexicon/new_voc_img_',
        ]
        pair_paths = [
            'evaluation/lexicons/ic13/GenericVocabulary_pair_list.txt',
            'evaluation/lexicons/ic13/ch2_test_vocabulary_pair_list.txt',
            'evaluation/lexicons/ic13/new_strong_lexicon/pair_voc_img_',
        ]
        lexicon_type = args.lexicon_type
    elif 'ic15' in args.result_path.lower():
        gt_folder = 'evaluation/gt/gt_ic15'; IS_WORDSPOTTING = False
        lexicon_paths = [
            'evaluation/lexicons/ic15/GenericVocabulary_new.txt',
            'evaluation/lexicons/ic15/ch4_test_vocabulary_new.txt',
            'evaluation/lexicons/ic15/new_strong_lexicon/new_voc_img_',
        ]
        pair_paths = [
            'evaluation/lexicons/ic15/GenericVocabulary_pair_list.txt',
            'evaluation/lexicons/ic15/ch4_test_vocabulary_pair_list.txt',
            'evaluation/lexicons/ic15/new_strong_lexicon/pair_voc_img_',
        ]
        lexicon_type = args.lexicon_type
    else:
        raise ValueError('Cannot determine target dataset')

    if args.with_lexicon:
        lexicon_path = lexicon_paths[lexicon_type]
        pair_path = pair_paths[lexicon_type]
        lexicons = read_lexicon(lexicon_path)
        pairs = read_pair(pair_path)
    else:
        lexicons = None; pairs = None

    print('Reading GT')
    gts = read_gt(gt_folder, IS_WORDSPOTTING)
    print('Reading and Processing Results')
    results = read_result(args.result_path, lexicons, pairs, 0.4, gt_folder, lexicon_type)

    print('Evaluating')
    conf_thres_list = np.arange(0.8, 0.95, 0.01)
    hmeans = []; recalls = []; precisions = []
    for conf_thres in conf_thres_list:
        precision, recall, hmean, pgt, ngt, ndet = evaluate(
            results=results,
            gts=gts,
            conf_thres=conf_thres,
        )
        hmeans.append(hmean); recalls.append(recall); precisions.append(precision)
    
    max_hmean = max(hmeans)
    max_hmean_index = len(hmeans) - hmeans[::-1].index(max_hmean) - 1
    precision = precisions[max_hmean_index]
    recall = recalls[max_hmean_index]
    conf_thres = conf_thres_list[max_hmean_index]
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Hmean: {max_hmean:.4f}, Conf Thres: {conf_thres:.4f}')

if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, help='Path to json result')
    parser.add_argument('--with_lexicon', action='store_true', help='Whether to evaluate with lexicons')
    parser.add_argument('--lexicon_type', type=int, choices=[0, 1, 2], default=0, help='0: Generic; 1: Weak; 2: Strong')
    args = parser.parse_args()

    main(args)