clear,clc,close all
warning off

load PaviaU
load PaviaU_gt

train = zeros(size(paviaU_gt));

for i = 1:9
    c = zeros(size(paviaU_gt));
    b = paviaU_gt == i;
    while 1
%         subplot(122)
        bw = roipoly(b);
        if isempty(bw) 
            break
        end
%         subplot(121)
%         imshow(bw.*b)
        c = c + bw.*b;  
%         title(['total = ',num2str(sum(sum(b))),'you selected = ',num2str(sum(sum(c)))])  
        [sum(sum(b)),sum(sum(c))]
        
    end    
    train = train + c*i;
end



test = zeros(size(paviaU_gt));
% 
% for i = 1:9
%     c = zeros(size(paviaU_gt));
%     b = (paviaU_gt == i) - (train == i);
%     while 1
% %         subplot(122)
%         bw = roipoly(b);
%         if isempty(bw) 
%             break
%         end
% %         subplot(121)
% %         imshow(bw.*b)
%         c = c + bw.*b;  
% %         title(['total = ',num2str(sum(sum(b))),'you selected = ',num2str(sum(sum(c)))])  
%         [sum(sum(b)),sum(sum(c))]
%         
%     end    
%     test = test + c*i;
% end


