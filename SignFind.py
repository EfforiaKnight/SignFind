import cv2
import numpy as np
import argparse
from rectselector import RectSelector


def align_images(img, ref, max_matches, good_match_percent):
    # Convert images to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_matches)
    keypoints_img, descriptors_img = orb.detectAndCompute(img_gray, None)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_img, descriptors_ref, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]

    # Draw top matches
    img_matches = cv2.drawMatches(img, keypoints_img, ref, keypoints_ref,
                                  matches, None)
    cv2.imwrite('matches.jpg', img_matches)

    # Extract location of good matches
    points_img = np.zeros((len(matches), 2), dtype=np.float32)
    points_ref = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points_img[i, :] = keypoints_img[match.queryIdx].pt
        points_ref[i, :] = keypoints_ref[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points_img, points_ref, cv2.RANSAC)

    # Use homography
    height, width, channels = ref.shape
    img_reg = cv2.warpPerspective(img, h, (width, height))

    return img_reg, h


def verify_signatures(img_ref, img_reg, coordinates):
    found_sig = False
    xmin, ymin, xmax, ymax = coordinates
    print('Verifying signatures...')
    print('Coordinates:\nymin={}:ymax={}\nxmin={}:xmax={}'.format(
        ymin, ymax, xmin, xmax))

    lower = [0, 100, 30]
    upper = [255, 200, 120]
    lower = np.array(lower, dtype='uint8')
    upper = np.array(upper, dtype='uint8')

    img_diff = cv2.subtract(img_ref, img_reg)
    img_diff = cv2.GaussianBlur(img_diff, (5, 5), 0)
    img_diff_hsv = cv2.cvtColor(img_diff, cv2.COLOR_BGR2HSV)
    img_diff_thresh = cv2.inRange(img_diff_hsv, lower, upper)

    if np.mean(img_diff_thresh[ymin:ymax, xmin:xmax]) > np.mean(img_diff_thresh):
        found_sig = True

    cv2.imwrite('img_diff.jpg', img_diff[ymin:ymax, xmin:xmax])
    cv2.imwrite('img_diff_thresh.jpg', img_diff_thresh[ymin:ymax, xmin:xmax])

    return found_sig


def parse_args():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-i', '--img', required=True, help='Path to the scanned image')
    ap.add_argument(
        '-r', '--img-ref', required=True, help='Path to the reference image')
    ap.add_argument(
        '--max-matches',
        default=1200,
        type=int,
        help='Max matches for ORB feature detector [default: 1200]')
    ap.add_argument(
        '--good-match-percent',
        default=0.45,
        type=float,
        help='Percent of good matches to keep [default: 0.45]')
    args = ap.parse_args()

    return args


def read_images(args):
    # Read reference image
    print('Reading reference image: ', args.img_ref)
    img_ref = cv2.imread(args.img_ref, cv2.IMREAD_COLOR)
    img_ref = cv2.resize(img_ref, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    # Read image to be aligned
    print('Reading image to align: ', args.img)
    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    return img, img_ref


def callback(rect):
    global coordinates
    coordinates = rect
    print('Aquired new coordinates of the signature')
    # xmin, ymin, xmax, ymax = coordinates
    # print('New coordinates:\nymin={}:ymax={}\nxmin={}:xmax={}'.format(
    #     ymin, ymax, xmin, xmax))


def main():
    global coordinates
    coordinates = [75, 870, 308, 955]

    # Parse command line arguments
    args = parse_args()

    # Read and resize images
    img, img_ref = read_images(args)

    print('Aligning images...')
    # Registered image will be restored in img_reg.
    # The estimated homography will be stored in h.
    img_reg, h = align_images(img, img_ref, args.max_matches,
                              args.good_match_percent)

    cv2.namedWindow('ImageRef')
    ROISelect = RectSelector('ImageRef', callback)
    cv2.imshow('ImageRef', img_ref)
    img_ref_copy = img_ref.copy()

    while 0xFF & cv2.waitKey(1) != 27:
        if ROISelect.dragging:
            img_ref_copy = img_ref.copy()
            ROISelect.draw(img_ref_copy)

        cv2.imshow('ImageRef', img_ref_copy)

    # Verify signatures
    found_sig = verify_signatures(img_ref, img_reg, coordinates)

    # Write aligned image to disk.
    out_filename = 'aligned.jpg'
    print('Saving aligned image: ', out_filename)
    cv2.imwrite(out_filename, img_reg)

    # Print estimated homography
    print('Estimated homography matrix: \n', h)

    print('=' * 30)
    if found_sig:
        print('Found Signature !!!')
    else:
        print('No Signature was found')
    print('=' * 30)


if __name__ == '__main__':
    main()
