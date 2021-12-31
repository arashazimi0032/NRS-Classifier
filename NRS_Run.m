

load PaviaU
load PaviaU_gt
load train600
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

lambda = [0.5]; 

class = NRS_Classification(X, n, Y, lambda);

class2 = class;
class3 = class;
b = zeros(1,size(image,1)*size(image,2));
for i =1:9
    c = test == i;
    c = c(:)';
    b(c) = class2(1:m(i));
    class2(1:m(i)) = [];
end
class = b;
toc
C = Classifier(paviaU,class);
final = ToRGB(C);
imshow(final)
[Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C);


