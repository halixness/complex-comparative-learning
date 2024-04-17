#%%
import torch
import clip

from torch.utils.data import DataLoader

from collections import OrderedDict
from config import *
from my_dataset import *
from my_models import *
from models.hypernetwork import HyperMem, count_parameters

import pickle
import argparse

from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

def get_batches(base_names, preprocessed_images_path, source):
	images = []
	for base_name in base_names:
		path = os.path.join(preprocessed_images_path, source, base_name+'_rgba.pickle')
		with open(path, 'rb') as file:
			emb = pickle.load(file)
			images.append(emb)
	images = torch.stack(images, dim = 0)
	return images

def my_clip_evaluation_base(model, in_path, preprocessed_images_path, source, in_base, types, dic, vocab):
    with torch.no_grad():

        # get dataset
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        data_loader = DataLoader(dt, batch_size=129, shuffle=True)

        top3 = 0
        top3_color = 0
        top3_material = 0
        top3_shape = 0
        tot_num = 0

        for base_is, names in data_loader:
            # Prepare the inputs
            images = get_batches(names, preprocessed_images_path, source)
            images = images.to(device)
            batch_size_i = len(base_is)
        
            ans = []
            # go through memory
            for label in vocabs:

				# compute stats
				z, centroid_i = model(label, images)
				z = z.squeeze(0)
				centroid_i = centroid_i.repeat(batch_size_i, 1)
				disi = ((z - centroid_i)**2).mean(dim=1)
                ans.append(disi.detach().to('cpu'))

            # get top3 incicies
            ans = torch.stack(ans, dim=1)
            values, indices = ans.topk(3, largest=False)
            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)

            # calculate stats
            tot_num += len(indices)
            for bi in range(len(indices)):
                ci = 0
                mi = 0
                si = 0
                if indices_lb[bi][0] in indices[bi]:
                    ci = 1
                if indices_lb[bi][1] in indices[bi]:
                    mi = 1
                if indices_lb[bi][2] in indices[bi]:
                    si = 1

                top3_color += ci
                top3_material += mi
                top3_shape += si

                if (ci == 1) and (mi == 1) and (si == 1):
                    top3 += 1

        print('BASIC: ','Num:',tot_num, 
        'Color:',top3_color / tot_num, 'Material:',top3_material / tot_num, 'Shape:',top3_shape / tot_num, 'Tot:',top3 / tot_num)

    return top3 / tot_num

def my_clip_evaluation_logical(model, in_path, preprocessed_images_path, source, in_base, types, dic, vocab):
    with torch.no_grad():

        # get dataset
        dt = MyDataset(in_path, source, in_base, types, dic, vocab)
        data_loader = DataLoader(dt, batch_size=129, shuffle=True)

        tot_num = 0
        score_and = 0
        tot_num_and = 0
        errors_and = dict()

        score_or = 0
        tot_num_or = 0

        score_not = 0
        tot_num_not = 0

        tot_num_logical = 0

        for base_is, names in data_loader:

            # Prepare the inputs
            images = get_batches(names, preprocessed_images_path, source)
            images = images.to(device)
            batch_size_i = len(base_is)

            ans_logical = []
            for label in logical_vocabs:

				# compute stats
				z, centroid_i = model(label, images)
				z = z.squeeze(0)
				centroid_i = centroid_i.repeat(batch_size_i, 1)
				disi = ((z - centroid_i)**2).mean(dim=1)
                ans_logical.append(disi.detach().to('cpu'))
            
            # get top3 incicies
            ans_logical = torch.stack(ans_logical, dim=1)
            values, indices = ans_logical.topk(106, largest=False) # 106 is the number of logical relations true for each image

            _, indices_lb = base_is.topk(3)
            indices_lb, _ = torch.sort(indices_lb)

            tot_num += len(indices)
            # calculate stats
            for bi in range(len(indices_lb)):
                # object
                color = vocabs[indices_lb[bi][0]]
                material = vocabs[indices_lb[bi][1]]
                shape = vocabs[indices_lb[bi][2]]
                atrs = [color, material, shape]

                # check logical rep retrieved
                for i in indices[bi]:
                    tot_num_logical += 1
                    # check validity
                    prop = logical_vocabs[i].split(' ')

                    if 'not' in prop:
                        attr1 = prop[1]
                        attr2 = None
                        tot_num_not += 1   

                        if attr1 not in atrs:
                            score_not += 1

                    elif 'and' in prop:
                        attr1 = prop[0]
                        attr2 = prop[2]
                        tot_num_and += 1

                        if attr1 in atrs and attr2 in atrs:
                            score_and += 1
                        else:
                            def get_attr(lesson):
                                for k,v in dic_test_logical.items():
                                    if lesson in v:
                                        return k 
                            attr1 = get_attr(attr1)
                            attr2 = get_attr(attr2)
                            if attr1+'_and_'+attr2 not in errors_and.keys():
                                errors_and[attr1+'_and_'+attr2] = 1
                            errors_and[attr1+'_and_'+attr2] += 1


                    elif 'or' in prop:
                        attr1 = prop[0]
                        attr2 = prop[2]
                        tot_num_or += 1

                        if attr1 in atrs or attr2 in atrs:
                            score_or += 1

            tot_score_logical = score_not + score_and + score_or
        
        print('LOGICAL: ','Num:',tot_num,'Tot:',tot_score_logical / tot_num_logical, 
        'Not:',score_not / tot_num_not, 'And:',score_and / tot_num_and, 'Or:',score_or / tot_num_or)
        print('AND errors:')
        pprint(errors_and)

    return tot_score_logical/tot_num_logical

# TESTING

types = ['rgba']
vocab = all_vocabs

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in_path', type=str, required=True)
    argparser.add_argument('--preprocessed_images_path', type=str, required=True)
    argparser.add_argument('--checkpoint', type=str, required=True)
    args = argparser.parse_args()

    clip_model, _ = clip.load("ViT-B/32", device=device)
    model = HyperMem(lm_dim=512, knob_dim=128, input_dim=512, hidden_dim=128, output_dim=latent_dim, clip_model=clip_model).to(device)
    
    # Adjusting weights: DDP -> single
    weights = torch.load(args.checkpoint)
    n_weights = OrderedDict()
    for k in weights.keys():
        newk = k.replace("module.", "")
        n_weights[newk] = weights[k]

    # Loading
    model.load_state_dict(n_weights)
    
    print('mare new obj')
    mare_new_obj = my_clip_evaluation_base(model, args.in_path, args.preprocessed_images_path, 'novel_test/', bn_n_test, types, dic_train, vocab)
    print('mare var')
    mare_var = my_clip_evaluation_base(model, args.in_path, args.preprocessed_images_path, 'test/', bn_test, types, dic_test, vocab)
    
    print('mare new obj')
    mare_logical_new_obj = my_clip_evaluation_logical(args.in_path, args.preprocessed_images_path, 'novel_test/', bn_n_test, types, dic_train_logical, vocab)
    print('mare var')
    mare_logical_var = my_clip_evaluation_logical(args.in_path, args.preprocessed_images_path, 'test/', bn_test, types, dic_test_logical, vocab)
