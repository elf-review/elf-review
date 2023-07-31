sample_numbers = 5
function_array = [None for i in range(df.shape[0])]
total_item = 0
for model_idx, model_item in df.iterrows():

    if total_item>10:
        break
    
    output_dict = {}
    output_dict[model_item['model_name']]={}
    output_dict[model_item['model_name']]['inputs'] = []
    output_dict[model_item['model_name']]['org_output'] = []
    output_dict[model_item['model_name']]['comp_output'] = []
    output_dict[model_item['model_name']]['comparison'] = []
    output_dict[model_item['model_name']]['task'] = model_item['category']
    local_model_path = 'models_comp/'+model_item['repo_link']+'/pytorch_model.bin'
    local_comp_model_path = 'models_compressed/'+model_item['repo_link']+'/pytorch_model.bin'
    try:
        loc_org_file = Path(local_model_path)
        loc_comp_file = Path(local_comp_model_path)
        if not loc_comp_file.exists() or hf_bucket.Object(model_item['compressed_model_path']).content_length != loc_comp_file.stat().st_size:
            hf_bucket.download_file(model_item['compressed_model_path'], local_comp_model_path)
        if not loc_org_file.exists() or hf_bucket.Object(model_item['original_model_path']).content_length != loc_org_file.stat().st_size:
            hf_bucket.download_file(model_item['original_model_path'], local_model_path)
        total_item += 1
    except ClientError:
        print('{} - compressed model does not exist'.format(model_item['model_name']))
        continue

    comp_model_pipeline = 'models_compressed/'+model_item['repo_link']
    org_model_pipeline =   'models_comp/'+model_item['repo_link']
    #os.remove(local_model_path)
    #os.remove(local_comp_model_path)

    match model_item['category']:
        case 'audio-classification':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(audio_classification_dataset['train'])-1)
                sample = audio_classification_dataset['train'][sample_id]['speech']['array']
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_audio_classification).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
            
        case 'automatic-speech-recognition':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(automatic_speech_recognitiondataset['train'])-1)
                sample = automatic_speech_recognitiondataset['train']['audio'][sample_id]['array']
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_automatic_speech_recognition).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'conversational':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(conversational_dataset['test'])-1)
                sample = conversational_dataset['test'][sample_id]['personas'][0]
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_conversational).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'feature-extraction':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(conversational_dataset['test'])-1)
                sample = conversational_dataset['test'][sample_id]['personas'][0]
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_feature_extraction).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'fill-mask':
            samples = []
            for i in range(sample_numbers):
                while True:
                    sample = random_nonascii_unicode_mask(alphabet)
                    if len(sample.split('[MASK]'))>1:
                        break
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_fill_mask).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'image-classification':
            samples = []
            for i in range(sample_numbers):
                sample = imagenet_list_dataset[random.randint(0,subsample_size-1)]['image']
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_image_classification).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'question-answering':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(dataset_qa)-1)
                sample = {'question':dataset_qa[sample_id]['question'], 'context':dataset_qa[sample_id]['context']}
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_question_answering).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'summarization':
            samples = []
            for i in range(sample_numbers):
                sample = random_nonascii_unicode_summarization(alphabet)
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_summarization).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'text-classification':
            samples = []
            for i in range(sample_numbers):
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_text_classification).remote , org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'text-generation':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(text_gen_dataset)-1)
                prompt_words = text_gen_dataset[sample_id]['prompt'].split(' ')
                prompt_len   = random.randint(2, len(prompt_words)-2)
                sample = ' '.join(prompt_words[0:prompt_len])
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_text_generation).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'text2text-generation':
            samples = []
            for i in range(sample_numbers):
                sample_id = random.randint(0, len(text_gen_dataset)-1)
                prompt_words = text_gen_dataset[sample_id]['prompt'].split(' ')
                prompt_len   = random.randint(2, len(prompt_words)-2)
                sample = ' '.join(prompt_words[0:prompt_len])
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_text2text_generation).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'token-classification':
            samples = []
            for i in range(sample_numbers):
                sample = random_nonascii_unicode_translation(alphabet)
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_token_classification).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'translation':
            samples = []
            for i in range(sample_numbers):
                sample = random_nonascii_unicode_translation(alphabet)
                samples.append(sample)
            function_array[model_idx] = [partial(ray.remote(task_translation).remote, org_model_pipeline, comp_model_pipeline, samples, output_dict[model_item['model_name']]), output_dict]
        case 'image-segmentation':
            pass
        case 'image-to-text':
            pass
        case 'universal-dependencies':
            pass
        case 'zero-shot-classification':
            pass
        case 'zero-shot-image-classification':
            pass
        case 'multiple-choice':
            pass
    

            