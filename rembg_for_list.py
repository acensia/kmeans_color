import os, glob
    
test_img = [f for f in glob.glob(os.path.join("./color_test_image","*.png")) if "test" in f]
    
    
def rembg_remove(img_paths, only_mask=False):
    import rembg
    from rembg import remove, new_session

    session = new_session()

    masked = ""
    if only_mask :
        os.makedirs('./rembg_mask', exist_ok=True)
        masked = "_mask"
    else :
        os.makedirs("./rembg", exist_ok=True)


    for f in img_paths:
        # org = cv2.imread(f)
        # out = remove(org,session=session)
        # cv2.imwrite(f'./rembg\\{f}_rembg', out)
        filename = os.path.basename(f)[:-4]

        with open(f, "rb") as i:
            with open(f'./rembg{masked}\\{filename}{masked}.png' , 'wb') as  o:
                input = i.read()
                output = remove(input, session=session, only_mask=only_mask)
                o.write(output)