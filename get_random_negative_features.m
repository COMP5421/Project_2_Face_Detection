function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

image_files = dir(fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
num_cells = feature_params.template_size / feature_params.hog_cell_size; % Number of hog cells in one template
D = num_cells^2 * 31; % Template dimensionality
samples_per_image = int32(num_samples / num_images); % Extract the same amount of negative samples from each image
features_neg = zeros(samples_per_image*num_images, D); % N by D matrix,avoid useless 0

% scales = [1, 0.9, 0.8, 0.7, 0.6];
% samples_per_scale = int32(num_samples / (num_images*5));

for i = 1 : num_images
    filename = [non_face_scn_path '/' image_files(i).name];
    img = imread(filename);
    [img_width, img_length, img_dim] = size(img);
    
    if img_dim == 3
        img = rgb2gray(img);
    end
    
    %img = histeq(img);
    
    % Solution 1
    % First extract region of interest,then compute HOG of the extracted region
    for j = 1 : samples_per_image
        x = int32(randi([1 img_width-feature_params.template_size+1]));
        y = int32(randi([1 img_length-feature_params.template_size+1]));
        window = img(x:x+feature_params.template_size-1, y:y+feature_params.template_size-1);
        hog = vl_hog(single(window), feature_params.hog_cell_size);
        hog = reshape(hog,[1, D]);
        features_neg((i-1)*samples_per_image+j,:) = hog;
    end 
    
    
    
    % Solution 2
    % Downsize image at multiple scales
%     q = 0;
%     for scal = scales
%         img_scaled = imresize(img, scal);
%         [img_scaled_width, img_scaled_length] = size(img_scaled);
%         HOG = vl_hog(single(img_scaled), feature_params.hog_cell_size);
%         if (img_scaled_width >= feature_params.template_size)&&(img_scaled_length >= feature_params.template_size)
%             for j = 1 : samples_per_scale
%                 x = int32(randi([0 img_scaled_width-feature_params.template_size]));
%                 y = int32(randi([0 img_scaled_length-feature_params.template_size]));
%                 %x = int32(rand*(img_width - feature_params.template_size));
%                 %y = int32(rand*(img_length - feature_params.template_size));
%                 hog = HOG((x/feature_params.hog_cell_size+1):(x/feature_params.hog_cell_size+num_cells),(y/feature_params.hog_cell_size+1):(y/feature_params.hog_cell_size+num_cells),:);
%                 hog = reshape(hog,[1, D]);
%                 features_neg((i-1)*samples_per_image+j+samples_per_scale*q,:) = hog;
%             end
%         end
%         q = q + 1;
%     end
end

end
