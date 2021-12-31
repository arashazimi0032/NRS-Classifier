clear,clc,close all

load PaviaU
load PaviaU_gt
load train1300
tic

image = paviaU;
N = sum(sum(train ~= 0));

test = paviaU_gt;

X =[];Y = [];
labels = [];
labels1 = [];
for i = 1:9
    ch = [];
    ch1 = [];
    c = train == i;
    c1 = test == i;
    n(i) = sum(sum(c));
    m(i) = sum(sum(c1));
    for j = 1:size(image,3)
        b = image(:,:,j);
        a = b(c);
        a1 = b(c1);
        ch(:,j) = a;
        ch1(:,j) = a1;
    end
    X = [X;ch];
    Y = [Y;ch1];
    labels = [labels;i*ones(n(i),1)];
    labels1 = [labels1;i*ones(m(i),1)];
end

% Y = zeros(size(image,1)*size(image,2),size(image,3));
% for j = 1:size(image,3)
%     b = image(:,:,j);
%     Y(:,j) = b(:);    
% end

% Normalize
Normalize = max(X(:));
X = X./Normalize;
Y = Y./Normalize;

% labels2 = randi([1,9],length(Y),1);

model = svmtrain(labels,X,'-t 2 -s 0 -c 150 -q -b 1');
% [predicted_label, accuracy,prob_estimates] = svmpredict( labels1,Y, model,'-b 1');
[predicted_label, accuracy,prob_estimates] = svmpredict( labels1,Y, model,'-b 1');

predicted_label2 = predicted_label;
b = zeros(1,size(image,1)*size(image,2));
for i =1:9
    c = test == i;
    c = c(:)';
    b(c) = predicted_label2(1:m(i));
    predicted_label2(1:m(i)) = [];
end
predicted_label = b;

toc
C = Classifier(paviaU,predicted_label');
final = ToRGB(C);
imshow(final)
[Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C);



