function features_pos = get_positive_features(train_path_pos, feature_params)
% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

image_files = dir(fullfile( train_path_pos, '*.jpg') ); % Caltech Faces stored as .jpg
num_images = length(image_files); % Number of faces/pictures -- N
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31; % Template dimensionality -- D
features_pos = zeros(num_images, D); % N by D matrix

for i = 1:num_images
    filename = [train_path_pos '/' image_files(i).name];
    img = imread(filename);
    %img_fliplr = fliplr(img);
    %img = histeq(img);
    hog = vl_hog(single(img), feature_params.hog_cell_size);
    %hog_fliplr = vl_hog(single(img_fliplr), feature_params.hog_cell_size);
    hog = reshape(hog, [1, D]);
    %hog_fliplr = reshape(hog_fliplr, [1, D]);
    features_pos(i, :) = hog;
    %features_pos(i+num_images, :) = hog_fliplr;
end

end
