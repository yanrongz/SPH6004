function index=find_this_matrix(a,b)
	%a�Ǿ���b��һ������
	%����������Ѱ�Ҿ���a��������b��ȵ���һ��
	k = size(a,1);
	i=0;
	for j = 1:k    %a�еĵ�j��
    	if a(j,:)==b
       		i = i+1;
       		index(i) = j;
    	end
	end
	index = index';    
end
