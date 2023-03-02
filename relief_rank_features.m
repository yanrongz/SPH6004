clear
data=xlsread('data_smo.csv');
[data_row,data_col] = size(data);

all_distance = cell(3,data_row);
	for j = 1:data_row
	    all_distance{1,j} = data(j,1:data_col-1);
	    middle_same = data(find(data(:,data_col)==data(j,data_col)),1:data_col-1); 
	    j_find = find_this_matrix(middle_same,data(j,1:data_col-1));
	    middle_same(j_find,:) = [];
	    distance = dist(data(j,1:data_col-1),middle_same);
	    min_index = find(distance==min(distance));
	    if length(min_index)>1
	        u = min_index(1);
	    else
	        u = min_index;
	    end
	    all_distance{2,j} = middle_same(u,:); 
        
	    middle_same = data(find(data(:,data_col)~=data(j,data_col)),1:data_col-1);
	    distance = dist(data(j,1:data_col-1),middle_same);
	    min_distance = find(distance==min(distance));
	    if length(min_distance)>1
	        u = min_distance(1);
	    else
	        u = min_distance;
	    end
	    all_distance{3,j} = middle_same(u,:);
    end
    
dota_matrix = zeros(data_row,data_col-1);
	for g = 1:data_row
	    dota_matrix(g,:) = -(all_distance{1,g}-all_distance{2,g}).^2+(all_distance{1,g}-all_distance{3,g}).^2;
	end
	dota = sum(dota_matrix,1);
number=[1:1:data_col-1];
dotanew=[number;dota];
sort_dotanew=sortrows(dotanew',2,'descend');

xlswrite('sort_score_index_R.csv',sort_dotanew);

    
    
    
    
    
    
    
    
    
    