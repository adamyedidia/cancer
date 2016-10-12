from PIL import Image
import os
from PIL import Image, ImageChops

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
        
def crop_to_max_square(img_file, outfile, target_square_length):
    im = Image.open(img_file)
    
    ####  remove the large black backgound in the image ########
    im=trim(im)

    input_width, input_height = im.size
    crop_rect_length = max(input_width, input_height)
    crop_rect = (max([input_width-crop_rect_length,0])/2, max([input_height-crop_rect_length, 0])/2, crop_rect_length, crop_rect_length)

    im = im.crop(crop_rect)
    target_size = (target_square_length, target_square_length)
    im.thumbnail(target_size)
    im.save(outfile, "JPEG")

def crop_all_images_to_max_square(image_dir, target_size):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_dir = os.path.join(dir_path, image_dir)

    for root, dirs, files in os.walk(full_dir):
        for name in files:
            if name.split('.')[-1] in {'jpg', 'png'}:
                full_path = os.path.join(root, name)
                parts = [el for el in full_path.split('/') if el]
                target_folder_path = '/{}-{}/{}/'.format('/'.join(parts[:-2]), target_size, parts[-2])
                target_full_path = '{}{}.jpg'.format(target_folder_path, parts[-1].split('.')[0])
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                crop_to_max_square(full_path, target_full_path, target_size)


# crop_all_images_to_max_square('train', 256)
# crop_all_images_to_max_square('dev', 256)

crop_all_images_to_max_square('train', 256)
crop_all_images_to_max_square('dev', 256)