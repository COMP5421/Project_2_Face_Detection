function [bboxes, confidences, image_ids] = run_detector_with_histequal(test_scn_path, w, b, feature_params)
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 

test_scenes = dir(fullfile(test_scn_path, '*.jpg'));
fprintf('~~~path: %s', test_scn_path);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

template_size = feature_params.template_size;
cell_size = feature_params.hog_cell_size;
num_cells = feature_params.template_size / feature_params.hog_cell_size; % Number of hog cells in one template
D = num_cells^2 * 31; % Template dimensionality
scales = [1.2, 1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
step = 3;

parfor i = 1:length(test_scenes)
	fprintf('Detecting faces in %s\n', test_scenes(i).name)
	img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    if(size(img, 3) == 3)
    	img = rgb2gray(img);
    end

    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0, 1);
    cur_image_ids = cell(0, 1);
    for scal = scales
        img_scaled = imresize(img, scal, 'bicubic');
        [height, width] = size(img_scaled);
        
        %HOG = vl_hog(single(img_scaled), cell_size);
   
            for j = 1:floor((height-template_size)/step)
                for k = 1:floor((width-template_size)/step)
        			window = img_scaled(1+(j-1)*step:(j-1)*step+template_size, 1+(k-1)*step:(k-1)*step+template_size);
                    window = histeq(window);
        			hog = vl_hog(single(window), cell_size);
                    %hog = HOG(j:j+num_cells-1,k:k+num_cells-1,:);
        			hog = reshape(hog, [1, D]);
        			conf = hog*w + b;
                    if conf > -0.5
                        temp_bboxes = [floor((1+(k-1)*step)/scal), floor((1+(j-1)*step)/scal), floor(((k-1)*step+template_size)/scal), floor(((j-1)*step+template_size)/scal)];
                        temp_confidences = conf;
        				temp_image_ids = test_scenes(i).name;
                        cur_bboxes = [cur_bboxes; temp_bboxes];
                        cur_confidences = [cur_confidences; temp_confidences];
                        cur_image_ids = [cur_image_ids; temp_image_ids];
                    end
                end
            end
    end
    
    % Avoid duplicate detections
    [is_valid] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    cur_confidences = cur_confidences(is_valid, :);
    cur_bboxes = cur_bboxes(is_valid, :);
    cur_image_ids = cur_image_ids(is_valid, :);
    bboxes = [bboxes; cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids = [image_ids; cur_image_ids];
end
end