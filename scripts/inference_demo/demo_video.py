from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
pretrained = "../../ckpt/DriveMM"
model_name = 'llama'  #get_model_name_from_path(pretrained)
device = torch.device('cuda:0')
llava_model_args = {
        "multimodal": True,
    }
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device, **llava_model_args)

model.eval()

'''lingoqa'''
urls = [['lingoqa/2a469a9042a47e4c68cadfaa7bdb4519/0.jpg', 'lingoqa/2a469a9042a47e4c68cadfaa7bdb4519/1.jpg', 'lingoqa/2a469a9042a47e4c68cadfaa7bdb4519/2.jpg', 'lingoqa/2a469a9042a47e4c68cadfaa7bdb4519/3.jpg', 'lingoqa/2a469a9042a47e4c68cadfaa7bdb4519/4.jpg']]
question = '<image>.\nThere is a video of traffic captured from the front view of the ego vehicle. What is the current action and its justification? Answer in the form \"action, justification\".'
urls = [['lingoqa/ab4845470b41f0e123da50c996c35745/0.jpg', 'lingoqa/ab4845470b41f0e123da50c996c35745/1.jpg', 'lingoqa/ab4845470b41f0e123da50c996c35745/2.jpg', 'lingoqa/ab4845470b41f0e123da50c996c35745/3.jpg', 'lingoqa/ab4845470b41f0e123da50c996c35745/4.jpg']]
question = '<image>.\nThere is a video of traffic captured from the front view of the ego vehicle. Is there a traffic light in the vicinity? If so, what color is it displaying?'
modalities=['video']

'''bdd-x'''
urls = [['bddx/1/testing_1f0fff77-a50aae97_23664_0.png', 'bddx/1/testing_1f0fff77-a50aae97_23664_3.png', 'bddx/1/testing_1f0fff77-a50aae97_23664_7.png', 'bddx/1/testing_1f0fff77-a50aae97_23664_11.png', 'bddx/1/testing_1f0fff77-a50aae97_23664_15.png']]
question = "<image>.\nThere is a video of traffic captured from the front view of the ego vehicle. Describe the current action of the ego car, and explain the cause of this car's action."
urls = [['bddx/2/testing_1f13b7b2-e98c7699_23665_0.png', 'bddx/2/testing_1f13b7b2-e98c7699_23665_3.png', 'bddx/2/testing_1f13b7b2-e98c7699_23665_7.png', 'bddx/2/testing_1f13b7b2-e98c7699_23665_11.png', 'bddx/2/testing_1f13b7b2-e98c7699_23665_15.png']]
question = "<image>.\nThere is a video of traffic captured from the front view of the ego vehicle. Describe the current action of the ego car, and explain the cause of this car's action."
modalities=['video']


image_tensors = []

images = []
for img_idx, cur_crls in enumerate(urls):
    cur_images = []
    for url in cur_crls:
        img_pil = Image.open(str(url)).convert("RGB")
        cur_images.append(img_pil)
    images.append(cur_images)
image_tensors = [process_images(cur_images, image_processor, model.config) for cur_images in images]
image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

conv_template = "llava_llama_3"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

if True:
    print("using train_prompt! note: now only support preprocess_llama3!!!")
    from llava.train.train import preprocess_llama3
    sources = [[{"from": 'human',"value": question},{"from": 'gpt', "value": ''}]]
    input_ids = preprocess_llama3(sources, tokenizer, has_image=True)['input_ids'][:, :-1].to(device)
else:
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

image_sizes = [[frame.size for frame in video] for video in image_tensors]

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=modalities
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
