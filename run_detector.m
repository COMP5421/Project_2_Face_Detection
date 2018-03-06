function [bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params)

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0, 4);
confidences = zeros(0, 1);
image_ids = cell(0, 1);

cell_size = feature_params.hog_cell_size;
template_size = feature_params.template_size;
D = (template_size / cell_size)^2 * 31;
scales = [1, 0.85, 0.75, 0.6, 0.5, 0.4, 0.25, 0.15, 0.1, 0.07];
step = 6;

for i = 1:length(test_scenes)
	fprintf('Detecting faces in %s\n', test_scenes(i).name)
	img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    if(size(img, 3) > 1)
    	img = rgb2gray(img);
    end
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0, 1);
    cur_image_ids = cell(0, 1);
    for scal = scales
        img_scaled = imresize(img, scal);
        [height, width] = size(img_scaled);
        if height>template_size && width>template_size
        	for j = 1:floor((height-template_size)/step)+1
        		for k = 1:floor((width-template_size)/step)+1
        			window = img_scaled(1+(j-1)*step:1+(j-1)*step+template_size-1, 1+(k-1)*step:1+(k-1)*step+template_size-1);
        			hog = vl_hog(single(window), cell_size);
        			hog = reshape(hog, [1, D]);
        			conf = hog*w+b;
        			if conf>1
                        temp_bboxes = [floor((1+(k-1)*step)/scal), floor((1+(j-1)*step)/scal), floor((1+(k-1)*step+template_size-1)/scal), floor((1+(j-1)*step+template_size-1)/scal)];
        				temp_confidences = conf;
        				temp_image_ids = test_scenes(i).name;
                        cur_bboxes = [cur_bboxes; temp_bboxes];
                        cur_confidences = [cur_confidences; temp_confidences];
                        cur_image_ids = [cur_image_ids; temp_image_ids];
                    end
                end
            end
        end
    end
    [is_valid] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));
    cur_confidences = cur_confidences(is_valid, :);
    cur_bboxes = cur_bboxes(is_valid, :);
    cur_image_ids = cur_image_ids(is_valid, :);
    bboxes = [bboxes; cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids = [image_ids; cur_image_ids];
end
end
