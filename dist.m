function distance=dist(a,b,p)
    if nargin==2 %ֻ����������ʱ
        p=2;
        distance=dist(a,b,p);
    end

    %�������������֮���ŷ�Ͼ���
    if size(a,2)~=size(b,2)
        disp('error:������������ͬ')
    else
        distance=zeros(1,size(b,1));
        for j=1:size(b,1)
            distance(j)=(sum((a-b(j,:)).^2))^(1/p); %��j��ÿһά��ƽ��֮�Ϳ�p�η�
        end
    end
end

