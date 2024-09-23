for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'read {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        cropped_faces, restored_faces, restored_imgs = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)
