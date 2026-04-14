# Author: Doby-Xu
# Source: WithAnyone/withanyone/id_loss_nofa.py
# Integrated into PixelSmile with minimal modifications.

from insightface.model_zoo import model_zoo

import torch
import numpy as np
from typing import List, Tuple

from PIL import Image
# import torchvision.transforms.functional as F
from torch.nn import functional as F


import skimage.transform as trans
import numpy as np

from scipy.optimize import linear_sum_assignment

# from info_nce import InfoNCE, info_nce

## Code from duonglong289: https://gist.github.com/duonglong289/79392e7062cbe8517294266ef7670703 ####################
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
arcface_src = torch.tensor(
    [[38.2946, 51.6963], 
     [73.5318, 51.5014], 
     [56.0252, 71.7366],
     [41.5493, 92.3655], 
     [70.7299, 92.2041]],
    dtype=torch.float32)

# Follow affine transform in skimage
def estimate_affine_torch(src, dst):
    n, d = src.shape
    _coeffs = range(d*d+1)
    
    src_matrix, src = _center_and_normalize_points_torch(src)
    dst_matrix, dst = _center_and_normalize_points_torch(dst)
    if not torch.all(torch.isfinite(src_matrix + dst_matrix)):
        params = torch.full((d + 1, d + 1), torch.nan)
        return params
    DEVICE = src.device
    # params: a0, a1, a2, b0, b1, b2, c0, c1
    A = torch.zeros((n * d, (d + 1) ** 2), device=DEVICE)
    # fill the A matrix with the appropriate block matrices; see docstring
    # for 2D example — this can be generalised to more blocks in the 3D and
    # higher-dimensional cases.
    for ddim in range(d):
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim * n : (ddim + 1) * n, ddim * (d + 1) + d] = 1
        A[ddim * n : (ddim + 1) * n, -d - 1 : -1] = src
        A[ddim * n : (ddim + 1) * n, -1] = -1
        A[ddim * n : (ddim + 1) * n, -d - 1 :] *= -dst[:, ddim : (ddim + 1)]

    
    # Select relevant columns, depending on params
    A = A[:, list(_coeffs) + [-1]]

    # Get the vectors that correspond to singular values, also applying
    # the weighting if provided
    _, _, V = torch.linalg.svd(A)

    # if the last element of the vector corresponding to the smallest
    # singular value is close to zero, this implies a degenerate case
    # because it is a rank-defective transform, which would map points
    # to a line rather than a plane.
    if torch.isclose(V[-1, -1], torch.tensor(0., device=DEVICE)):
        params = torch.full((d + 1, d + 1), torch.nan)
        return params

    H = torch.zeros((d + 1, d + 1), device=DEVICE)
    # solution is right singular vector that corresponds to smallest
    # singular value
    H.view(-1)[list(_coeffs)] = -V[-1, :-1] / V[-1, -1]
    # H.view(-1)[-1] = H.view(-1)[0]
    H[d, d] = 1

    # De-center and de-normalize
    H = torch.linalg.inv(dst_matrix) @ H @ src_matrix

    # Small errors can creep in if points are not exact, causing the last
    # element of H to deviate from unity. Correct for that here.
    H /= H[-1, -1].clone()

    params = H
    return params

def _center_and_normalize_points_torch(points):
    n, d = points.shape
    DEVICE = points.device
    centroid = torch.mean(points, axis=0)

    centered = points - centroid
    rms = torch.sqrt(torch.sum(centered**2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return torch.full((d + 1, d + 1), torch.nan), torch.full_like(points, torch.nan)

    norm_factor = torch.sqrt(torch.tensor(d, device=DEVICE)) / rms
    part_matrix = norm_factor * torch.concat(
        (torch.eye(d, device=DEVICE), -centroid[:, None]), axis=1)
    matrix = torch.concat(
        (
            part_matrix, torch.tensor([[0,] * d + [1]], device=DEVICE),
        ),
        axis=0,
    )

    points_h = torch.vstack([points.T, torch.ones(n, device=DEVICE)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points

def align_face(img: torch.Tensor, landmark: torch.Tensor, image_size: int=112):
    '''
    img: (H,W,C) - full image
    landmark: shape(5,2) - facial landmarks in full image
    '''
    
    img_hei, img_wid = img.shape[:2]

    device = img.device
    float_dtype = img.dtype
    
    src = landmark.to(device=device, dtype=float_dtype)
    dst = arcface_src.to(device=device, dtype=float_dtype)
    
    src = src / torch.tensor([img_wid, img_hei], dtype=float_dtype, device=device) * 2 - 1
    dst = dst / torch.tensor([image_size, image_size], dtype=float_dtype, device=device) * 2 - 1

    theta = estimate_affine_torch(dst, src)
    theta.unsqueeze_(0)
    
    # Process image tensor
    # default_float_dtype = torch.get_default_dtype()
    default_float_dtype = float_dtype
    img_tensor = img.permute((2,0,1)).contiguous()
    # img_tensor.div_(255.)
    # img_tensor.unsqueeze_(0)
    # the above two in-place operations cut the gradient flow, we need to use the following instead
    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    output_size = torch.Size((1, 3, image_size, image_size))
    grid = F.affine_grid(theta[:, :2], output_size).to(device=device, dtype=default_float_dtype)
    img_tensor = F.grid_sample(img_tensor, grid, align_corners=False, mode='bicubic')
    # img_tensor.squeeze_(0)
    img_tensor = img_tensor.squeeze(0)
    
    aligned_img = img_tensor.permute((1,2,0))*255
    
    return aligned_img
###############################################################################

def estimate_norm_torch(lmk, image_size=112, mode='arcface', device=None):
    """
    PyTorch version of estimate_norm function
    
    Args:
        lmk (torch.Tensor): 5 facial landmarks of shape (5, 2)
        image_size (int): Output image size
        mode (str): Alignment mode
        device: Device to place tensors on
        
    Returns:
        torch.Tensor: Transformation matrix of shape (2, 3)
    """
    # Convert to tensor if not already
    if not isinstance(lmk, torch.Tensor):
        lmk = torch.tensor(lmk, dtype=torch.float32, device=device)
    else:
        device = lmk.device if device is None else device
        lmk = lmk.to(device)
    
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    
    # Use arcface_dst as a tensor
    arcface_dst_torch = torch.tensor(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=torch.float32, device=device)
    
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    
    dst = arcface_dst_torch * ratio
    dst[:, 0] += diff_x
    
    # Convert to numpy for transform estimation (required by scikit-image)
    lmk_np = lmk.detach().cpu().numpy()
    dst_np = dst.detach().cpu().numpy()
    
    tform = trans.SimilarityTransform()
    tform.estimate(lmk_np, dst_np)
    M_np = tform.params[0:2, :]
    
    # Convert back to tensor
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    
    return M

class DET:
    def __init__(self, det_model_path):
        self.model_det = model_zoo.get_model(det_model_path,providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model_det.prepare(ctx_id=0, det_thresh=0.4, input_size=(640, 640))

    def __call__(self, image):

        image = image.clone().detach().cpu().to(torch.float32).numpy().transpose(1, 2, 0)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        bboxes, kpss = self.model_det.detect(image)
        return bboxes, kpss


def norm_crop_torch(img, landmark, image_size=112, mode='arcface'):
    """
    PyTorch version of norm_crop function using torch.nn.functional.grid_sample
    
    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W)
        landmark (torch.Tensor or numpy.ndarray): 5 facial landmarks of shape (5, 2)
        image_size (int): Output image size
        mode (str): Alignment mode
        
    Returns:
        torch.Tensor: Aligned face image tensor of shape (C, image_size, image_size)
    """
    # Determine device
    device = img.device if isinstance(img, torch.Tensor) else None
    
    # Ensure img is a tensor with correct shape
    if not isinstance(img, torch.Tensor):
        if isinstance(img, np.ndarray):
            # Convert HWC to CHW if needed
            if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
                img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float()
        else:
            # Handle other image types
            from torchvision import transforms
            img = transforms.ToTensor()(img)
    
    if device is not None:
        img = img.to(device)
        
    # Add batch dimension if not present
    need_squeeze = False
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        need_squeeze = True
    
    # Get transformation matrix
    M = estimate_norm_torch(landmark, image_size, mode, device=device)
    
    # Get image dimensions
    batch_size, channels, height, width = img.shape
    
    # Convert to PyTorch affine grid format
    # Create full 3x3 matrix
    M_full = torch.eye(3, device=device)
    M_full[:2, :] = M
    
    # Create the source and destination normalization matrices
    src_norm = torch.tensor([
        [2.0/width, 0, -1],
        [0, 2.0/height, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    dst_norm = torch.tensor([
        [2.0/image_size, 0, -1],
        [0, 2.0/image_size, -1],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Calculate the transformation matrix for grid_sample
    theta = src_norm @ torch.inverse(M_full) @ torch.inverse(dst_norm)
    theta = theta[:2, :].unsqueeze(0)
    
    # Create the sampling grid
    grid = torch.nn.functional.affine_grid(theta, [batch_size, channels, image_size, image_size], align_corners=False)
    
    # grid to image dtype
    grid = grid.type_as(img)
    # Apply the transformation
    warped = torch.nn.functional.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # Remove batch dimension if added
    if need_squeeze:
        warped = warped.squeeze(0)
    
    return warped

def detect_face_pose(landmarks, threshold=0.78):
    """
    Determine if a face is front-facing or side-facing based on landmarks.
    
    Args:
        landmarks: 5-point facial landmarks [left_eye, right_eye, nose, left_mouth, right_mouth]
        threshold: Threshold for determining front vs side face (0.0-1.0)
    
    Returns:
        "front", "side", or "profile" string indicating the face pose
    """
    # Landmarks typically in order: left_eye, right_eye, nose, left_mouth, right_mouth
    left_eye, right_eye = landmarks[0], landmarks[1]
    
    # Calculate eye distance ratio (horizontal)
    left_eye_x, right_eye_x = left_eye[0], right_eye[0]
    eye_distance = abs(right_eye_x - left_eye_x)
    
    # Calculate face width from bounding box of landmarks
    face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
    
    # Normalized eye distance ratio (percentage of face width)
    eye_distance_ratio = eye_distance / (face_width + 1e-6)
    
    # Check eye horizontal symmetry
    if eye_distance_ratio > threshold:
        return "front"
    elif eye_distance_ratio > threshold * 0.5:
        return "partial_profile"
    else:
        return "profile"
    
REF_CLUSTER_CENTER = ""
import os
class IDLoss:
    def __init__(self, device='cuda', use_state_negative_pool=False, det_model_path=None, rec_model_path=None):
        self.device = device
        self.netArc = torch.load(
            rec_model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        self.netArc = self.netArc.to(self.device, dtype=torch.bfloat16)
        self.netArc.eval()
        self.netArc.requires_grad_(True)
        self.netDet = DET(det_model_path) # an onnx model however
        self.dtype = torch.bfloat16

        # Initialize the negative pool
        if use_state_negative_pool:
            self._init_negative_pool()

    def _init_negative_pool(self):
        """
        Load all npy embeddings from REF_CLUSTER_CENTER, and store them in a num_tensor, 512 tensors
        """
        self.negative_pool = []
        if os.path.exists(REF_CLUSTER_CENTER):
            for npy_file in os.listdir(REF_CLUSTER_CENTER):
                if npy_file.endswith('.npy'):
                    npy_path = os.path.join(REF_CLUSTER_CENTER, npy_file)
                    try:
                        embedding = np.load(npy_path, allow_pickle=True)
                        if isinstance(embedding, np.ndarray):
                            if embedding.ndim == 2 and embedding.shape[1] == 512:
                                self.negative_pool.append(torch.tensor(embedding, dtype=self.dtype, device=self.device))
                            elif len(embedding) == 512:
                                self.negative_pool.append(torch.tensor(embedding, dtype=self.dtype, device=self.device))
                    except:
                        pass
                
        if len(self.negative_pool) > 0:
            self.negative_pool = torch.stack(self.negative_pool, dim=0)
        else:
            self.negative_pool = None

    def get_arcface_embeddings(self, images, check_side_views=False, original_bboxes=None):
        """
        Get ArcFace embeddings for a batch of images, assuming single person per image.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            List of embeddings (one tensor [1, 512] per image).
            Boolean indicating if face alignment was forced.
        """
        self.netArc.eval()
        self.netArc.requires_grad_(True)
        
        if len(images.shape) == 3: 
            images = images.unsqueeze(0) 
        
        batch_size = images.shape[0]
        all_embeddings = []
        force_align_any = False

        for batch_idx in range(batch_size):
            image = images[batch_idx]
            force_align = False
            
            with torch.no_grad():
                bboxes, landmarks = self.netDet(image)
                if check_side_views:
                    if len(landmarks) > 0:
                        if detect_face_pose(landmarks[0]) != "front":
                            force_align = True
                    if any((bbox[2] - bbox[0] < 80 or bbox[3] - bbox[1] < 80) for bbox in bboxes):
                        force_align = True

            # Use detection result or fallback (single face assumption)
            if len(landmarks) == 0:
                landmark = arcface_src
                force_align = True
            else:
                # Always take the first detected face aka the most prominent one
                landmark = torch.tensor(landmarks[0], dtype=torch.float32)

            if force_align:
                force_align_any = True
            
            # Align face
            aimg = align_face(image.permute(1, 2, 0), landmark).permute(2, 0, 1) # (C, H, W)
            aimg = aimg.unsqueeze(0).to(dtype=torch.float32)
            aimg = (aimg - 127.5) / 127.5
            
            embedding = self.netArc(aimg.to(dtype=self.dtype)) # [1, 512]
            all_embeddings.append(embedding)

        return all_embeddings, force_align_any
    
    
    def compute_id_loss(self, decoded_images, ground_truth_arcface_embeddings, rec_bbox_A=None, rec_bbox_B=None, filter_out_side_views=False, original_bboxes=None):
        """
        Compute identity loss (1 - cosine similarity) for single person per image.
        """
        generated_arcface_embeddings, force_align = self.get_arcface_embeddings(decoded_images, check_side_views=filter_out_side_views)
        
        if not generated_arcface_embeddings:
            return None, True

        id_losses = []

        for i in range(len(generated_arcface_embeddings)):
            gen_emb = generated_arcface_embeddings[i]                 # [1, 512]
            gt_emb = ground_truth_arcface_embeddings[i]               # Expect [1, 512] or [512]
            
            # Normalize GT shape
            if gt_emb.ndim == 1:
                gt_emb = gt_emb.unsqueeze(0)
            if gt_emb.shape[0] > 1:
                # If GT has multiple faces, take the first one
                gt_emb = gt_emb[0:1]
            
            cossim = torch.nn.functional.cosine_similarity(gen_emb, gt_emb)
            id_losses.append(1 - cossim)

        id_loss = torch.mean(torch.stack(id_losses))
        
        return id_loss, force_align
    
    def compute_id_loss_with_embeddings(self, generated_arcface_embeddings, ground_truth_arcface_embeddings, rec_bbox_A=None, rec_bbox_B=None, filter_out_side_views=False, force_align=False, original_bboxes=None):
        """
        Compute identity loss directly from embeddings.
        """
        id_losses = []

        for i in range(len(generated_arcface_embeddings)):
            gen_emb = generated_arcface_embeddings[i]
            gt_emb = ground_truth_arcface_embeddings[i]
            
            # Handle list wrapper or multi-face structures from previous pipeline stages
            if isinstance(gen_emb, list): gen_emb = gen_emb[0]
            if isinstance(gt_emb, list): gt_emb = gt_emb[0]
            
            if gen_emb.ndim == 1: gen_emb = gen_emb.unsqueeze(0)
            if gt_emb.ndim == 1: gt_emb = gt_emb.unsqueeze(0)
            
            if gen_emb.shape[0] > 1: gen_emb = gen_emb[0:1]
            if gt_emb.shape[0] > 1: gt_emb = gt_emb[0:1]

            cossim = torch.nn.functional.cosine_similarity(gen_emb, gt_emb)
            id_losses.append(1 - cossim)

        id_loss = torch.mean(torch.stack(id_losses))
        return id_loss
    
    def compute_id_loss_two_images(self, images1, images2):
        """
        Compute identity loss between two batches of images.
        Args:
            images1: Batch of images (B, C, H, W)
            images2: Batch of images (B, C, H, W)
        Returns:
            id_loss: Mean Identity loss (1 - cosine similarity)
        """
        embeddings1, _ = self.get_arcface_embeddings(images1)
        embeddings2, _ = self.get_arcface_embeddings(images2)
        
        if not embeddings1 or not embeddings2:
            return None

        # Ensure we compare valid pairs
        min_len = min(len(embeddings1), len(embeddings2))
        
        id_losses = []
        for i in range(min_len):
            emb1 = embeddings1[i]
            emb2 = embeddings2[i]
            
            # Normalize shapes
            if emb1.ndim == 1: emb1 = emb1.unsqueeze(0)
            if emb2.ndim == 1: emb2 = emb2.unsqueeze(0)
            
            cossim = torch.nn.functional.cosine_similarity(emb1, emb2)
            id_losses.append(1 - cossim)
            
        if not id_losses:
            return None
            
        id_loss = torch.mean(torch.stack(id_losses))
        return id_loss

    # def compute_contrastive_loss(self, generated_embeddings, ground_truth_embeddings, 
    #                           generated_labels=None, ground_truth_labels=None, 
    #                           temperature=0.07, max_negatives=10):
    #     """
    #     InfoNCE for single face. Matches generated[i] with ground_truth[i].
    #     Uses other ground truths in batch as negatives.
    #     """
    #     device = self.device
    #     dtype = self.dtype
    
    #     gen = generated_embeddings.to(device=device, dtype=dtype)
    #     gt = ground_truth_embeddings.to(device=device, dtype=dtype)
        
    #     # Flatten extra dimensions (B, 1, D) -> (B, D)
    #     if gen.ndim > 2: gen = gen.view(gen.shape[0], -1)
    #     if gt.ndim > 2: gt = gt.view(gt.shape[0], -1)

    #     gen = F.normalize(gen, dim=1)
    #     gt  = F.normalize(gt, dim=1)
        
    #     B = gen.shape[0]
        
    #     # Use simple symmetric cross-entropy for batch contrastive
    #     # Similarity between each gen and all GTs
    #     logits = torch.matmul(gen, gt.T) / temperature # [B, B]
    #     labels = torch.arange(B, device=device) # [0, 1, ... B-1]
        
    #     loss = F.cross_entropy(logits, labels)
    #     return loss
    
    # def compute_info_nce_loss(self, generated_embeddings, ground_truth_embeddings, extend_negative_pool=None):
    #     device = self.device
    #     dtype = self.dtype
    
    #     gen = generated_embeddings.to(device=device, dtype=dtype)
    #     gt = ground_truth_embeddings.to(device=device, dtype=dtype)
        
    #     if gen.ndim > 2: gen = gen.view(gen.shape[0], -1)
    #     if gt.ndim > 2: gt = gt.view(gt.shape[0], -1)
        
    #     gen = F.normalize(gen, dim=1) # [B, D]
    #     gt  = F.normalize(gt, dim=1)  # [B, D]
        
    #     loss_fn = InfoNCE(negative_mode='paired')

    #     query = gen
    #     positive_keys = gt
        
    #     # Create negatives from batch
    #     B = gen.shape[0]
    #     negative_keys = []
    #     for i in range(B):
    #         # Use all valid GTs except i as negatives
    #         indices = torch.arange(B, device=device) != i
    #         neg_keys = gt[indices]
    #         negative_keys.append(neg_keys)
            
    #     negative_keys = torch.stack(negative_keys, dim=0) # [B, B-1, D]

    #     if extend_negative_pool is not None:
    #         ext_neg = extend_negative_pool.to(device=device, dtype=dtype)
    #         # Flatten if needed
    #         if ext_neg.ndim == 4:
    #             ext_neg = ext_neg.view(ext_neg.shape[0], -1, ext_neg.shape[-1])
    #         negative_keys = torch.cat([negative_keys, ext_neg], dim=1)

    #     return loss_fn(query, positive_keys, negative_keys)

    # def region_diffusion_loss(self, decoded_images, ground_truth_image, bboxes_A, bboxes_B=None, 
    #                                weights_A=1.0, weights_B=1.0, background_weight=-1,
    #                                loss_type='mse', normalize_by_area=True):
    #     '''
    #     Optimized regional diffusion loss for face restoration
    #     - Supports weighted loss for different regions
    #     - Optional background loss with lower weight
    #     - Multiple loss function options (MSE, L1, SSIM)
    #     - Normalization by region area to avoid bias toward larger regions
    #     '''
    #     # 1. Get batch dimensions
    #     b, c, h, w = decoded_images.shape
    #     loss_list = []
    #     background_losses = []

    #     decoded_images = decoded_images.float()
    #     ground_truth_image = ground_truth_image.float()
        
    #     # If weights are scalar, expand to list for each batch item
    #     if not isinstance(weights_A, (list, tuple)):
    #         weights_A = [weights_A] * b
    #     if bboxes_B is not None and not isinstance(weights_B, (list, tuple)):
    #         weights_B = [weights_B] * b
        
    #     # Create full image mask for background calculation
    #     if background_weight > 0:
    #         full_mask = torch.ones((b, 1, h, w), device=decoded_images.device)
        
    #     for i in range(b):
    #         # Create region masks for current image
    #         region_masks = []
    #         region_weights = []
            
    #         # Handle A bounding box
    #         if bboxes_A is not None:
    #             bbox_A = bboxes_A[i]
    #             # Ensure bbox is within image boundaries
    #             # print("bbox_A", bbox_A)
    #             # exit()
    #             y1_A, x1_A = max(0, int(bbox_A[1])), max(0, int(bbox_A[0]))
    #             y2_A, x2_A = min(h, int(bbox_A[3])), min(w, int(bbox_A[2]))
                
    #             if y2_A > y1_A and x2_A > x1_A:  # Valid bbox
    #                 # Crop regions
    #                 decoded_region_A = decoded_images[i, :, y1_A:y2_A, x1_A:x2_A]
    #                 ground_truth_region_A = ground_truth_image[i, :, y1_A:y2_A, x1_A:x2_A]
                    
    #                 # Calculate loss based on specified type
    #                 if loss_type == 'mse':
    #                     region_loss_A = torch.nn.functional.mse_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
    #                 elif loss_type == 'l1':
    #                     region_loss_A = torch.nn.functional.l1_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
    #                 elif loss_type == 'smooth_l1':
    #                     region_loss_A = torch.nn.functional.smooth_l1_loss(decoded_region_A, ground_truth_region_A, reduction='mean')
                    
    #                 # Normalize by area if requested
    #                 if normalize_by_area:
    #                     area_A = (y2_A - y1_A) * (x2_A - x1_A)
    #                     region_loss_A = region_loss_A * (h * w / area_A) if area_A > 0 else region_loss_A
                    
    #                 loss_list.append(weights_A[i] * region_loss_A)
                    
    #                 # Create mask for background calculation
    #                 if background_weight > 0:
    #                     mask_A = torch.zeros((1, h, w), device=decoded_images.device)
    #                     mask_A[:, y1_A:y2_A, x1_A:x2_A] = 1
    #                     region_masks.append(mask_A)
            
    #         # Handle B bounding box if provided
    #         if bboxes_B is not None:
    #             bbox_B = bboxes_B[i]
    #             # Ensure bbox is within image boundaries
    #             y1_B, x1_B = max(0, int(bbox_B[1])), max(0, int(bbox_B[0]))
    #             y2_B, x2_B = min(h, int(bbox_B[3])), min(w, int(bbox_B[2]))
                
    #             if y2_B > y1_B and x2_B > x1_B:  # Valid bbox
    #                 # Crop regions
    #                 decoded_region_B = decoded_images[i, :, y1_B:y2_B, x1_B:x2_B]
    #                 ground_truth_region_B = ground_truth_image[i, :, y1_B:y2_B, x1_B:x2_B]
                    
    #                 # Calculate loss based on specified type
    #                 if loss_type == 'mse':
    #                     region_loss_B = torch.nn.functional.mse_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
    #                 elif loss_type == 'l1':
    #                     region_loss_B = torch.nn.functional.l1_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
    #                 elif loss_type == 'smooth_l1':
    #                     region_loss_B = torch.nn.functional.smooth_l1_loss(decoded_region_B, ground_truth_region_B, reduction='mean')
                    
    #                 # Normalize by area if requested
    #                 if normalize_by_area:
    #                     area_B = (y2_B - y1_B) * (x2_B - x1_B)
    #                     region_loss_B = region_loss_B * (h * w / area_B) if area_B > 0 else region_loss_B
                    
    #                 loss_list.append(weights_B[i] * region_loss_B)
                    
    #                 # Create mask for background calculation
    #                 if background_weight > 0:
    #                     mask_B = torch.zeros((1, h, w), device=decoded_images.device)
    #                     mask_B[:, y1_B:y2_B, x1_B:x2_B] = 1
    #                     region_masks.append(mask_B)
            
    #         # Calculate background loss if enabled
    #         if background_weight > 0 and region_masks:
    #             # Combine all region masks
    #             combined_mask = torch.clamp(torch.sum(torch.stack(region_masks), dim=0), 0, 1)
    #             # Invert to get background mask
    #             background_mask = 1 - combined_mask
                
    #             # Apply mask to get background pixels only
    #             decoded_bg = decoded_images[i] * background_mask
    #             ground_truth_bg = ground_truth_image[i] * background_mask
                
    #             # Calculate background loss
    #             if torch.sum(background_mask) > 0:  # Only if there are background pixels
    #                 if loss_type == 'mse':
    #                     bg_loss = F.mse_loss(decoded_bg, ground_truth_bg, reduction='sum') / (torch.sum(background_mask) + 1e-8)
    #                 elif loss_type == 'l1':
    #                     bg_loss = F.l1_loss(decoded_bg, ground_truth_bg, reduction='sum') / (torch.sum(background_mask) + 1e-8)
    #                 elif loss_type == 'smooth_l1':
    #                     bg_loss = F.smooth_l1_loss(decoded_bg, ground_truth_bg, reduction='sum') / (torch.sum(background_mask) + 1e-8)
                    
    #                 background_losses.append(background_weight * bg_loss)
        
    #     # Combine all losses

    #     all_losses = loss_list + background_losses
    #     if all_losses:
    #         total_loss = torch.mean(torch.stack(all_losses))
    #     else:
    #         # Fallback to full image loss if no valid regions
    #         total_loss = torch.nn.functional.mse_loss(decoded_images, ground_truth_image)
            
    #     return total_loss

    def get_arcface_embeddings_with_features(self, images, expected_num_faces=None):
        """
        Get ArcFace embeddings and hidden features for single face.
        """
        self.netArc.eval()
        self.netArc.requires_grad_(True)
        
        target_module = None
        for name, module in self.netArc.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and "251" in name:
                target_module = module
                break
        if target_module is None:
            for name, module in self.netArc.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) and module.num_features == 512:
                    target_module = module
                    break
        
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        batch_size = images.shape[0]
        all_embeddings = []
        all_hidden_features = []
        all_bboxes = []
        force_align_any = False
    
        for batch_idx in range(batch_size):
            image = images[batch_idx]
            force_align = False
            
            with torch.no_grad():
                bboxes, landmarks = self.netDet(image)
            
            all_bboxes.append(bboxes)
            
            if len(landmarks) == 0:
                landmark = arcface_src
                force_align = True
            else:
                landmark = torch.tensor(landmarks[0], dtype=torch.float32).to(device=image.device)
            
            if force_align:
                force_align_any = True

            features_holder = []
            def hook_fn(module, input, output):
                features_holder.append(output.detach().clone())
            
            hook = None
            if target_module is not None:
                hook = target_module.register_forward_hook(hook_fn)
            
            try:
                aimg = align_face(image.permute(1, 2, 0), landmark).permute(2, 0, 1)
                aimg = aimg.unsqueeze(0).to(dtype=torch.float32)
                aimg = (aimg - 127.5) / 127.5
                
                embedding = self.netArc(aimg.to(dtype=self.dtype))
                all_embeddings.append(embedding) # [1, 512]
                
                if features_holder:
                    all_hidden_features.append(features_holder[0]) # [1, 512, 7, 7]
                else:
                    all_hidden_features.append(torch.zeros(1, 512, 7, 7, device=embedding.device, dtype=embedding.dtype))
            finally:
                if hook is not None:
                    hook.remove()

        all_hidden_features = torch.stack(all_hidden_features, dim=0)
        
        return all_embeddings, all_hidden_features, force_align_any, all_bboxes
    
    def __call__(self, decoded_images, ground_truth_arcface_embeddings, ground_truth_image, bboxes_A, bboxes_B=None, regional_mse_weight = 3):
        """
        Compute the ID loss and regional diffusion loss.
        
        Args:
            decoded_images: Decoded images from the model
            ground_truth_arcface_embeddings: Ground truth arcface embeddings
            ground_truth_image: Ground truth image tensor
            bboxes_A: Bounding boxes for region A
            bboxes_B: Bounding boxes for region B (optional)
            regional_mse_weight: Weight for regional MSE loss
            
        Returns:
            id_loss: Identity loss
            regional_loss: Regional diffusion loss
        """
        # Compute ID loss
        id_loss = self.compute_id_loss(decoded_images, ground_truth_arcface_embeddings)
        
        # Compute regional diffusion loss
        if regional_mse_weight > 0:
            regional_loss = self.region_diffusion_loss(decoded_images, ground_truth_image, bboxes_A, bboxes_B=bboxes_B)
        
            return id_loss + regional_mse_weight * regional_loss
        
import random

def single_face_preserving_resize(img, face_bbox, target_size=512):
    """
    Resize image while ensuring a single face is preserved in the output.
    
    Args:
        img: PIL Image
        face_bbox: Single [x1, y1, x2, y2] face coordinates
        target_size: Maximum dimension for resizing
        
    Returns:
        Resized image that maintains the face, or None if face can't fit
    """
    # Extract face coordinates
    x1, y1, x2, y2 = map(int, face_bbox)
    
    # If any coordinates are negative, we cannot resize
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return None

    # Check if face dimensions fit in image
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width > img.height or face_height > img.width:
        return None

    # Choose cropping strategy based on image aspect ratio
    if img.width > img.height:
        # Crop width to make a square
        square_size = img.height
        
        # Calculate valid horizontal crop range that preserves the face
        left_max = x1  # Leftmost position that includes face left edge
        right_min = x2 - square_size  # Rightmost position that includes face right edge
        
        if right_min <= left_max:
            # We can find a valid crop window
            start = random.randint(int(right_min), int(left_max)) if right_min < left_max else int(right_min)
            start = max(0, min(start, img.width - square_size))  # Ensure within image bounds
        else:
            # Face is too wide for square crop - use center of face
            face_center = (x1 + x2) // 2
            start = max(0, min(face_center - (square_size // 2), img.width - square_size))
        
        cropped_img = img.crop((start, 0, start + square_size, square_size))
    else:
        # Crop height to make a square
        square_size = img.width
        
        # Calculate valid vertical crop range that preserves the face
        top_max = y1  # Topmost position that includes face top edge
        bottom_min = y2 - square_size  # Bottommost position that includes face bottom edge
        
        if bottom_min <= top_max:
            # We can find a valid crop window
            start = random.randint(int(bottom_min), int(top_max)) if bottom_min < top_max else int(bottom_min)
            start = max(0, min(start, img.height - square_size))  # Ensure within image bounds
        else:
            # Face is too tall for square crop - use center of face
            face_center = (y1 + y2) // 2
            start = max(0, min(face_center - (square_size // 2), img.height - square_size))
        
        cropped_img = img.crop((0, start, square_size, start + square_size))
    
    # Final resize to target size
    cropped_img = cropped_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return cropped_img
    
if __name__ == "__main__":
    # a test for the IDLoss class
    from PIL import Image
    import insightface
    import cv2
    # from ..refloader import extract_moref
    import os
    import json
    
    
    # randomly pick 2 jpg images from /data/data/2person/v1/ref_untar_1/Angelababy
    image_paths = [
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg",
        "/data/MIBM/BenCon/benchmarks/v200/untar_cn/0bb9e7100a32c4324bbd87e38ed0a11dba8876c0.jpg"
    ]
    images = []
    images_np = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        # img = np.array(img)
        json_dict = json.load(open(image_path.replace(".jpg", ".json")))
        img_face = single_face_preserving_resize(img, json_dict["bboxes"][0], )
        # save it for debug
        img_face.save("debug_face.jpg")
        img_face = np.array(img_face) 

        images.append(torch.tensor(img_face, dtype=torch.float32).permute(2, 0, 1))  # Convert to (C, H, W)
        # convert np img to bgr for opencv
        images_np.append(cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR))

    id_loss_model = IDLoss()
    official_model = insightface.app.FaceAnalysis(name = "antelopev2", root="./models", providers=['CUDAExecutionProvider'])
    official_model.prepare(ctx_id=0, det_thresh=0.4)
    
    arcface_embeddings_1_official = official_model.get(images_np[0])
    arcface_embeddings_2_official = official_model.get(images_np[1])
    # to tensor
    arcface_embeddings_1_official = torch.tensor(arcface_embeddings_1_official[0].embedding, dtype=torch.float32).unsqueeze(0)
    arcface_embeddings_2_official = torch.tensor(arcface_embeddings_2_official[0].embedding, dtype=torch.float32).unsqueeze(0)
    print(f"Official embeddings shapes: {arcface_embeddings_1_official.shape}, {arcface_embeddings_2_official.shape}")
    # cos
    cos_sim = torch.nn.functional.cosine_similarity(arcface_embeddings_1_official, arcface_embeddings_2_official)
    print("Cosine similarity between official embeddings:", cos_sim.item())


    images = torch.stack(images).to(id_loss_model.device, dtype=torch.float32)  # Convert to (B, C, H, W)
    print("Images shape:", images.shape)
    embeddings = id_loss_model.get_arcface_embeddings(images) # Embeddings shapes: [torch.Size([1, 512]), torch.Size([1, 512])]
    print("Embeddings shapes:", [emb[0].shape for emb in embeddings])
    cos_ours = torch.nn.functional.cosine_similarity(embeddings[0][0], embeddings[1][0])
    print("Cosine similarity between our embeddings:", cos_ours.item())
    cos_AA = torch.nn.functional.cosine_similarity(embeddings[0][0].to("cpu"), arcface_embeddings_1_official)
    cos_BB = torch.nn.functional.cosine_similarity(embeddings[1][0].to("cpu"), arcface_embeddings_2_official)
    print("Cosine similarity between our embeddings and official embeddings:", cos_AA.item(), cos_BB.item())

    # test a predicted image and a ground truth image
    gt_embeddings = np.load("/data/MIBM/UNO/tmp/ff6497adbba76f22a7fe6f377f0176a2eb2c25a3.npy",allow_pickle=True).item()["embeddings"]

    pred_image = Image.open("/data/MIBM/UNO/debug_output.png").convert("RGB")
    pred_image = np.array(pred_image)
    pred_image = torch.tensor(pred_image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
    pred_image = pred_image.unsqueeze(0).to(id_loss_model.device, dtype=torch.float32)  # Convert to (1, C, H, W)
    print("Predicted image shape:", pred_image.shape)
    # calculate id loss
    id_loss = id_loss_model.compute_id_loss(pred_image, [torch.tensor(gt_embeddings, dtype=torch.float32).to(id_loss_model.device)])
    print("ID Loss:", id_loss.item())
