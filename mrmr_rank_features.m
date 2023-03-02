clear
data=xlsread('data_smo.csv');
[data_row,data_col] = size(data);
x_train=data(:,1:data_col-1);
y_train=data(:,data_col-1);
fea=mrmr_miq_d(x_train,y_train,78);
xlswrite('sort_score_index_mrmr.csv',fea);