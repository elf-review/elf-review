file information

-- ray_fuzzy.py (run fuzzy testing on all models)
	-- the list of working models is in models_repo_info_fuzzy_600.csv

-- simple_fuzzy.py
	-- fuzzy testing without ray or parallelization
	-- should be run after fuzzy_ray.py as some models do not runs on ray

-- fuzzy_text_generation.py
	-- does fuzzy testing for only text generation models
	-- can be considered as a separate module

-- popularity_hf.ipynb
	-- calculates the trend of all models on Huggingface, their number of downloads, likes and
		repo change/commit history and when it was first uploaded

-- model_list_hf.pkl
	-- contains the meta data of models on Huggingface calculated through popularity_hf.ipynb

-- script_hf.py
	-- extract the information from model_list_hf.pkl and put it in a nice dictionary or pandas 		format that is easy to plot
	-- requires HuggingFace auth_token to run on all the models

-- convert_ascii_clean.csv
	-- contains the ascii values of tokens that are used for fuzzy testing of language models

-- requirements.txt
	-- huggingface libraries required to do fuzzy testing

-- not_working_models_new (a csv file)
	-- contains the models that were not fuzzy tested as their pipeline failed to execute

-- models_repo_info_fuzzy_600.csv
	-- list of models that are used for fuzzy testing
	-- not every model in it can be fuzzy tested
	-- these models were selected based on the models that were used for the compression analysis

-- debug_ray.ipynb
	-- to debug the ray module for fuzzy testing

-- all the parameters required for fuzzy testing are in the simple_fuzzy.py and ray_fuzzy.py. The script is automated and does not require user 
