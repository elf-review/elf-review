import pickle
from tqdm import tqdm
from huggingface_hub.hf_api import repo_info
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.hf_api import list_models
from huggingface_hub.hf_api import list_repo_commits
from huggingface_hub import login
from huggingface_hub.utils._errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError

#model_list =list(list_models())

#with open('model_list_hf.pkl', 'wb') as f:
#    pickle.dump(model_list, f)

auth_token = ''
login(auth_token)

with open('model_list_hf.pkl', 'rb') as f:
    model_list = pickle.load(f)

model_data = []
for idx, model in tqdm(enumerate(model_list), total=len(model_list)):
    try:
        model = model.id
        model_data.append({'repo_info':repo_info(model, files_metadata=True), 'commit_info': list_repo_commits(model), 'status':'working'})
    except (GatedRepoError, RepositoryNotFoundError,HfHubHTTPError) as e:
        model_data.append({'repo_info':None, 'commit_info': None, 'status':e})
    if idx%1000==0:
        with open('model_infor_hf.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    if idx%50000==0:
        with open(f'model_infor_hf_{idx}.pkl', 'wb') as f:
            pickle.dump(model_data, f)

with open('model_infor_hf.pkl', 'wb') as f:
    pickle.dump(model_data, f)