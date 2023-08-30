import os, glob
import rembg
from rembg import remove, new_session
    
def rembg_remove(img_paths, only_mask=True, image_type="*.png"):
    
    test_img = glob.glob(os.path.join(img_paths,image_type))

    session = new_session()

    masked = ""
    if only_mask :
        os.makedirs(f"{img_paths}_mask", exist_ok=True)
        masked = "_mask"
    else :
        os.makedirs(f"./{img_paths}_masked", exist_ok=True)
        masked = "_masked"


    for f in test_img:
        # org = cv2.imread(f)
        # out = remove(org,session=session)
        # cv2.imwrite(f'./rembg\\{f}_rembg', out)
        filename = os.path.basename(f)[:-4]
        # print(filename)
        with open(f, "rb") as i:
            with open(f'{img_paths}{masked}\\{filename}{masked}.png' , 'wb') as  o:
                input = i.read()
                output = remove(input, session=session, only_mask=only_mask)
                o.write(output)
                
    return f"{img_paths}{masked}"