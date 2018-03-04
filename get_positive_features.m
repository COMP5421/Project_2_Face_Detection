function [features_pos, num_images] = get_positive_features(train_path_pos, feature_params)

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;

features_pos = rand(num_images, D);

for i = 1:num_images
    filename = [train_path_pos '/' image_files(i).name];
    img = imread(filename);
    hog = vl_hog(single(img), feature_params.hog_cell_size, 'verbose');
    hog = reshape(hog, [1, D]);
    features_pos(i, :) = hog;
end
end