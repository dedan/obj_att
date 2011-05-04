% super simple watershed tests to get some more experience

clear 
close all

% first load my selfpainted test image
ss = imread('../images/super_simple.jpg');
ss = max(ss(:)) - ss;

% do watershed on original image
L = watershed(ss);
Lrgb = label2rgb(L);
figure(1)
subplot 211
imshow(ss)
subplot 212
imshow(Lrgb)
% --> das macht keinen sinn. ich muss es also wirklich auf dem
% gradientenbild machen

% do watershed on gradient magnitude
Iy = imfilter(double(ss), fspecial('sobel'), 'replicate');
Ix = imfilter(double(ss), fspecial('sobel')', 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
L = watershed(gradmag);
Lrgb = label2rgb(L);
figure(2)
subplot 211
imshow(gradmag)
subplot 212
imshow(Lrgb)

% mark the objects
figure(3)
subplot 211
fgm = imextendedmax(ss, 10);
imagesc(fgm)
subplot 212
bgm = imextendedmin(ss, 10);
imagesc(bgm)



gradmag2 = imimposemin(gradmag, fgm);
L = watershed(gradmag2);
Lrgb = label2rgb(L);
figure(4)
subplot 211
imshow(gradmag2)
subplot 212
imshow(Lrgb)

% not enought, maybe I have to mark the background


D = bwdist(fgm);
DL = watershed(D);
fgm1 = DL == 0;
figure, imshow(fgm1), title('Watershed ridge lines (bgm)')


gradmag3 = imimposemin(gradmag, fgm | fgm1);
L = watershed(gradmag3);
Lrgb = label2rgb(L);
figure
subplot 211
imshow(gradmag3)
subplot 212
imshow(Lrgb)
