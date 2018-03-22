function [new_negative_hog] = MineHardNegatives(test_scn_path, w, b, feature_params)
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
fprintf('~~~path: %s', test_scn_path);

%initialize these as empty and incrementally expand them.

template_size = feature_params.template_size;
cell_size = feature_params.hog_cell_size;
num_cells = feature_params.template_size / feature_params.hog_cell_size; % Number of hog cells in one template
D = (template_size / cell_size)^2 * 31; % Template dimensionality
%scales = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
scales = [1.2, 1.15, 1.1, 1.05, 1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1];
step = 3;
new_negative_hog = zeros(1, D);

for i = 1:length(test_scenes)
	fprintf('Detecting faces in %s\n', test_scenes(i).name)
	img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    if(size(img, 3) == 3)
    	img = rgb2gray(img);
    end
    %img = histeq(img);
    for scal = scales
        img_scaled = imresize(img, scal);
        [height, width] = size(img_scaled);
        
        HOG = vl_hog(single(img_scaled), cell_size);       
        for j = 1:floor((height-template_size)/step)
            for k = 1:floor((width-template_size)/step)
                %window = img_scaled(1+(j-1)*step:(j-1)*step+template_size, 1+(k-1)*step:(k-1)*step+template_size);
                %hog = vl_hog(single(window), cell_size);
                hog = HOG(j:j+num_cells-1,k:k+num_cells-1,:);
                hog = reshape(hog, [1, D]);
                conf = hog*w + b;
                if conf > 0.75
                    new_negative_hog = [new_negative_hog;hog];
                end
            end
        end
    end
end


end
