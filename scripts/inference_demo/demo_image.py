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
model_name = 'llama'  
device = torch.device('cuda:0')
llava_model_args = {
        "multimodal": True,
    }
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device, **llava_model_args)

model.eval()

'''codalm'''
urls = ['codalm/codalm1.png']
question = "<image>\nThere is an image of traffic captured from the front view of the ego vehicle. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
urls = ['codalm/codalm2.png']
question = "<image>\nThere is an image of traffic captured from the front view of the ego vehicle. Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
urls = ['codalm/codalm3.png']
question = "<image>\nThere is an image of traffic captured from the front view of the ego vehicle. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."
modalities=['image']


'''drivelm'''
urls = ['drivelm/n008-2018-08-30-10-33-52-0400__CAM_FRONT__1535639717612404.jpg', 'drivelm/n008-2018-08-30-10-33-52-0400__CAM_FRONT_LEFT__1535639717604799.jpg', 'drivelm/n008-2018-08-30-10-33-52-0400__CAM_FRONT_RIGHT__1535639717620482.jpg', 'drivelm/n008-2018-08-30-10-33-52-0400__CAM_BACK__1535639717637558.jpg', 'drivelm/n008-2018-08-30-10-33-52-0400__CAM_BACK_LEFT__1535639717647405.jpg', 'drivelm/n008-2018-08-30-10-33-52-0400__CAM_BACK_RIGHT__1535639717628113.jpg']
question = '1: <image> 2: <image> 3: <image> 4: <image> 5: <image> 6: <image>. These six images are the front view, front left view, front right view, back view, back left view and back right view of the ego vehicle. What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.'
question = '1: <image> 2: <image> 3: <image> 4: <image> 5: <image> 6: <image>. These six images are the front view, front left view, front right view, back view, back left view and back right view of the ego vehicle. Would <c3,CAM_FRONT,373.7,526.2> be in the moving direction of the ego vehicle?'
modalities=['image', 'image', 'image', 'image', 'image', 'image']

images = [Image.open(str(url)).convert("RGB") for url in urls]
image_tensors = process_images(images, image_processor, model.config)
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
            
image_sizes = [image.size for image in images]



# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=modalities,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])