clear,clc,close all

load PaviaU

image = paviaU;



name = {'train600','train700','train800','train900','train1300'};
for ii = 1:length(name)
    clearvars -except image paviaU overall_acKNN overall_acSVM overall_acNRS x ii name
    load PaviaU_gt
    load(name{ii})
    test = paviaU_gt;
    N = sum(sum(train ~= 0));
    
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
    
    
    % Normalize
    Normalize = max(X(:));
    X = X./Normalize;
    Y = Y./Normalize;
    
    
    %----------------------------%
    %KNN----
    
    class = knnclassify(Y, X, labels,3);
    
    class2 = class;
    b = zeros(1,size(image,1)*size(image,2));
    for i =1:9
        c = test == i;
        c = c(:)';
        b(c) = class2(1:m(i));
        class2(1:m(i)) = [];
    end
    class = b;
    
    
    C = Classifier(paviaU,class');
    final = ToRGB(C);
    figure,imshow(final)
    [Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C);
    overall_acKNN(ii) = overall_ac;
    %-------------------------------
    %SVM----
    if ii == 3
        model = svmtrain(labels,X,'-t 2 -s 0 -c 150 -q -b 1');
    else
        model = svmtrain(labels,X,'-t 2 -s 0 -c 1000 -q -b 1');
    end
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
    
    
    C = Classifier(paviaU,predicted_label');
    final = ToRGB(C);
    imshow(final)
    [Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C);
    overall_acSVM(ii) = overall_ac;
    %-----------------------------------
    %NRS---
%     
%     lambda = [0.5];
%     
%     class = NRS_Classification(X, n, Y, lambda);
%     
%     class2 = class;
%     b = zeros(1,size(image,1)*size(image,2));
%     for i =1:9
%         c = test == i;
%         c = c(:)';
%         b(c) = class2(1:m(i));
%         class2(1:m(i)) = [];
%     end
%     class = b;
%     
%     C = Classifier(paviaU,class);
%     final = ToRGB(C);
%     imshow(final)
%     [Confiusion_Matrix,overall_ac,user_ac,prod_ac] = ConfiusionMatrix(test,C);
%     overall_acNRS(ii) = overall_ac;
    
    x(ii) = sum(sum(train ~= 0));
    
    
end

% load overall_acKNN
load overall_acNRS

% overall_acKNN(end - 1) = (overall_acKNN(end)+overall_acKNN(end - 2))/2;
plot(x,overall_acKNN,'b-.s','markerfacecolor','blue','markersize',8)
hold on
plot(x,overall_acSVM,'r-.^','markerfacecolor','red','markersize',8)
hold on
plot(x,overall_acNRS,'k-.^','markerfacecolor','black','markersize',8)
grid on
xlabel('Number of Training Samples')
ylabel('Overall Accuracu')
legend('KNN','SVM','NRS','Location','southeast')