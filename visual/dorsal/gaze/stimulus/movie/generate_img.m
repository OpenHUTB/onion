run('../../../config.m');

gray_screen_hdv720(HEIGHT, WIDTH, 3) = 96;
gray_screen_hdv720 = uint8(gray_screen_hdv720);
gray_screen_hdv720(:, :, :) = ones(HEIGHT, WIDTH, 3)*96;
% imshow(gray_screen_hdv720);

imwrite(gray_screen_hdv720, fullfile(movie_dir, 'gray_screen_hdv720.png') );

