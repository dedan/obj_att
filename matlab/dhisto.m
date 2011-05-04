
clear

I = load('../images/img_depth1.txt');
max_dist = max(max(I));
%I(I == 0) = max_dist;
I(I > 3000) = 0;

figure(1)
subplot 211
imagesc(I)
[n, xout] = hist(I(:), 100:10:3000);
n = conv(n, ones(1,5) ./ 5);
subplot 212
bar(n(10:end));
figure(2)
imagesc(I)

c = 1;
out = zeros(size(I));
last = 2;
first = 2;
for i = 49:length(n)-5
    
    disp('');
    disp(n(i));
    disp(n(last))
    disp(n(first))
    
    % wenn der aktuelle wert groesser ist als der letzte, dann merk ihn dir
    if n(i) > n(last)
        last = i;
    end
    
    
    % wenn der aktuelle wert kleiner
    if n(i) < n(first)
        first = i;
    end
    
    % wenn wir am ende eines tales angekommen sind
    if n(i) < 0.4 * n(last)
        out(I > xout(first) & I < xout(i)) = c;
        first = i;
        last = i;
        c = c+1;
    end
end

imagesc(label2rgb(out))

figure(3)
hist(out(:))
