import subprocess

# List of commands with arguments
commands = [
    # [
    #     'python', 'scripts/pose2seg_confidence.py',
    #     '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
    #     '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
    #     '--GT-is-with-vis',
    #     '--use-bbox',
    #     '--no-update-bboxes',
    #     '--selection-method', 'distance+confidence',
    #     '--num-pos-keypoints', '2',
    #     '--num-neg-keypoints', '1',
    #     '--out-filename', '../BMP_results/SAM_ablation/g13',
    #     # '--expand-bbox',
    #     # '--oracle',
    #     # '--bbox-by-iou', '0.5',
    #     # '--num-pos-keypoints-if-bbox', '6',
    # ],
    # [
    #     'python', 'scripts/pose2seg_confidence.py',
    #     '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
    #     '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
    #     '--GT-is-with-vis',
    #     '--use-bbox',
    #     '--no-update-bboxes',
    #     '--selection-method', 'distance+confidence',
    #     '--num-pos-keypoints', '2',
    #     '--num-neg-keypoints', '3',
    #     '--out-filename', '../BMP_results/SAM_ablation/g14',
    #     # '--expand-bbox',
    #     # '--oracle',
    #     # '--bbox-by-iou', '0.5',
    #     # '--num-pos-keypoints-if-bbox', '6',
    # ],

    # [
    #     'python', 'scripts/pose2seg_confidence.py',
    #     '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
    #     '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
    #     '--GT-is-with-vis',
    #     '--use-bbox',
    #     '--no-update-bboxes',
    #     '--selection-method', 'distance+confidence',
    #     '--num-pos-keypoints', '2',
    #     '--num-neg-keypoints', '5',
    #     '--out-filename', '../BMP_results/SAM_ablation/g15',
    #     # '--expand-bbox',
    #     # '--oracle',
    #     # '--bbox-by-iou', '0.5',
    #     # '--num-pos-keypoints-if-bbox', '6',
    # ],
    # [
    #     'python', 'scripts/pose2seg_confidence.py',
    #     '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
    #     '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
    #     '--GT-is-with-vis',
    #     '--no-use-bbox',
    #     '--no-update-bboxes',
    #     '--selection-method', 'distance+confidence',
    #     '--num-pos-keypoints', '4',
    #     '--num-neg-keypoints', '1',
    #     '--out-filename', '../BMP_results/SAM_ablation/g16',
    #     # '--expand-bbox',
    #     # '--oracle',
    #     # '--bbox-by-iou', '0.5',
    #     # '--num-pos-keypoints-if-bbox', '6',
    # ],

    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
        '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--GT-is-with-vis',
        '--no-use-bbox',
        '--no-update-bboxes',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '4',
        '--num-neg-keypoints', '3',
        '--out-filename', '../BMP_results/SAM_ablation/g17',
        # '--expand-bbox',
        # '--oracle',
        # '--bbox-by-iou', '0.5',
        # '--num-pos-keypoints-if-bbox', '6',
    ],
    [
        'python', 'scripts/pose2seg_confidence.py',
        '--dataset', 'OCHuman', '--subset', 'val', '--gt-file', '../data/OCHuman/COCO-like/annotations/person_keypoints_val2017.json',
        '--num-images', '-1', '--no-debug-vis', '--output-as-list', 
        '--GT-is-with-vis',
        '--no-use-bbox',
        '--no-update-bboxes',
        '--selection-method', 'distance+confidence',
        '--num-pos-keypoints', '4',
        '--num-neg-keypoints', '5',
        '--out-filename', '../BMP_results/SAM_ablation/g18',
        # '--expand-bbox',
        # '--oracle',
        # '--bbox-by-iou', '0.5',
        # '--num-pos-keypoints-if-bbox', '6',
    ],


]

# Run commands sequentially
for command in commands:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with return code {e.returncode}")
    except Exception as e:
        print(f"An error occurred while running command '{command}': {str(e)}")