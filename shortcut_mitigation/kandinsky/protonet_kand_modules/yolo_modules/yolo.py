import torch
import matplotlib.pyplot as plt



# * Run YOLO on batches of Kand subimages and crop out the detected primitives.
def yolo_detect_and_crop_primitives_batch(batch_sub_imgs, concept_extractor, resize_transform, args, expected_crops=3, max_attempts=10):
    B = batch_sub_imgs.shape[0]
    cropped_per_image = []
    
    # Loop through each image individually
    for i in range(B):
        sub_img = batch_sub_imgs[i]
        attempt = 0
        crops = []
        while attempt < max_attempts and len(crops) < expected_crops:
            # Run detection on the individual image.
            results = concept_extractor.predict(sub_img.unsqueeze(0), 
                                                verbose=False, 
                                                device='cuda:'+args.GPU_ID,
                                                name=args.yolo_folder,
                                                project=args.yolo_folder)
            crops = []
            for result in results:
                boxes = sorted(result.boxes.xyxy, key=lambda box: (box[0]**2 + box[1]**2)**0.5)
                crop_count = 0
                for box in boxes:
                    if crop_count >= expected_crops:
                        break
                    # Get box coordinates (assuming they’re in the coordinate system of sub_img)
                    x1, y1, x2, y2 = map(int, box)
                    _, H, W = sub_img.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if (x2 - x1) > 0 and (y2 - y1) > 0:
                        crop = sub_img[:, y1:y2, x1:x2]
                        crop_resized = resize_transform(crop)
                        crops.append(crop_resized)
                        crop_count += 1
            attempt += 1
            if args.debug:
                print(f"Attempt {attempt} detected {len(crops)} primitives.")

        # Pad if needed.
        if len(crops) < expected_crops:
            for _ in range(expected_crops - len(crops)):
                crops.append(torch.zeros(3, 64, 64, device=batch_sub_imgs.device))
        cropped_per_image.append(crops)
    
    # Flatten and stack the crops.
    flat_crops = [crop for crops in cropped_per_image for crop in crops]
    assert len(flat_crops) == B * expected_crops and all(crop.shape == (3, 64, 64) for crop in flat_crops), \
        f"Expected shape {(B * expected_crops, 3, 64, 64)}, but got {[crop.shape for crop in flat_crops]}"
    
    if args.debug:
        plot_cropped(torch.stack(flat_crops))

    return torch.stack(flat_crops)


# * Plot the first 9 images in cropped_batch.
def plot_cropped(cropped_batch):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < cropped_batch.shape[0]:
            # Convert the tensor to a numpy array and transpose to (H, W, C) for plotting.
            img = cropped_batch[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
    plt.tight_layout()
    plt.show()