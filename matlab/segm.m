
clear
close all

%% Step 1: load the depth image, invert and show
I = load('../images/img_depth1.txt');
I(I == 0) = max(max(I));
imagesc(I)
I = max(max(I)) - I;
figure(1)
subplot 241
imagesc(I)
colormap gray
% mark the objects
subplot 242
fgm = imextendedmax(I, 500);
imagesc(fgm)


% compute gradient magnitude
Iy = imfilter(double(I), fspecial('sobel'), 'replicate');
Ix = imfilter(double(I), fspecial('sobel')', 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
subplot 245
imagesc(gradmag)

D = bwdist(fgm);
DL = watershed(D);
bgm = DL == 0;
subplot 246
imagesc(bgm)

gradmag3 = imimposemin(gradmag, fgm | bgm);
L = watershed(gradmag3);
Lrgb = label2rgb(L);
subplot(2,4, [3 4])
imagesc(gradmag3)
subplot(2,4, [7 8])
imagesc(Lrgb)

%%
figure
imagesc(imread('../images/img_rgb2.jpg'));
