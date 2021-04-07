clc
clear
tic

% parameters=[5 0.5 5 0.0001 1000 1];%% DNM 参数k q M ita epoch MiniBatchSize
parameters=[5 0.5 6 100 1000]; % k qs M popsize epoch
%%Dividedataset
%% COVID dataset
load India;
price = India;
DivideBoundary = 300; %%训练集大小
n = length(price);
L=2;% dimensions
r=1;% time delay
Dend = n-(L-1)*r-1; % Dend-divideBoundary= 测试集大小
%% preprocess
price_n = zeros(L+1, n-(L-1)*r-1);
for i=1:n-(L-1)*r-1
    price_n(1:L,i) = price(i:r:i+(L-1)*r);
    price_n(L+1,i) = price(i+(L-1)*r+1);
end
    trainx = price_n(1:L, 1:DivideBoundary); % Training Data
    trainy = price_n(L+1, 1:DivideBoundary);
    testx = price_n(1:L, DivideBoundary+1:Dend); % Testing Data
    testy = price_n(L+1, DivideBoundary+1:Dend);
%%归一化
[trainx_normalized, st1] = mapminmax(trainx,0.3,0.65);
[trainy_normalized, st2] = mapminmax(trainy,0.3,0.65);
testx_normalized = mapminmax('apply',testx,st1);  % Make "Testing data" be normalized just the same as that with "Training data".
testy_normalized = mapminmax('apply',testy,st2);

% for i =1:t_times
%     
% % [O1,Error,E, w,q]=D_Model_online(trainx_normalized, trainy_normalized, parameters);
% [O1, Error, E, w, q, Sample2, MiniFeatrueBatch] = D_Model_Train(trainx_normalized, trainy_normalized, parameters);
% [O2,~,~,accuracy] = D_Model_Prediction(testx_normalized, testy_normalized,w,q,parameters,Sample2,MiniFeatrueBatch);
% % [accuracy2, Threshold] = D_model_Prediction_logic_circuit(testx_normalized, testy_normalized, w,q,parameters(3));
%  Acc1(i)=accuracy;
% %  Acc2(i)=accuracy2;
[O Error w q w2] = SFSMS_func(trainx_normalized, trainy_normalized, parameters);
trainy_predicated = mapminmax('reverse', O, st2);
[testy_prediction_nor, Error_prediction, E_prediction] = D_Model_Prediction(testx_normalized, testy_normalized, w, q, w2, parameters);
testy_prediction = mapminmax('reverse', testy_prediction_nor, st2);

%% Display the Final Simulation Results
figure(1)
rawdata=[trainy testy];
newdata=[trainy_predicated' testy_prediction];
% rawdata=[trainy_normalized testy_normalized];
% newdata=[O(:,MaxEpoch)'];
x=1:length(rawdata);
x_bounary = length(trainy);
% 
plot(x,rawdata,'b-');
hold on
plot(x,newdata,'r--')
B=get(gca, 'YLim');
plot([x_bounary, x_bounary], [B(1), B(2)], '--m');
legend('actual value','predictive value','Location','NorthWest');
title('训练及预测图','FontName','Times New Roman','FontWeight','Bold','FontSize',12);
xlabel('Time','FontName','Times New Roman','FontSize',14);
ylabel('The values of InFlow','FontName','Times New Roman','FontSize',14);
%text('Units','pixels','Position',[40,40],'String','MATLAB');
text('Position',[x_bounary*2/3,B(2)*5/6],'String','Training','color','r');%
text('Position',[x_bounary + 5,B(2)*5/6],'String','Prediction','color','r');%
hold off

%%绘制收敛曲线
figure(2);
plot(Error);
legend('Mean Squred Error of Training Phase');
title('误差计算','FontName','Times New Roman','FontWeight','Bold','FontSize',12);
xlabel('Learning Epoch','FontName','Times New Roman','FontSize',14);
ylabel('MSE','FontName','Times New Roman','FontSize',14);

Mse = sum(((testy-testy_prediction).^2))/length(testy);
Rmse= (sum(abs((testy-testy_prediction).^2))/length(testy)).^0.5;
Mae= sum(abs((testy-testy_prediction)))/length(testy);
Mape= sum(abs((testy-testy_prediction)./testy))/length(testy);


disp(['MSE:',num2str(Mse)]);
disp(['RMSE',num2str(Rmse)]);
disp(['MAE:',num2str(MAE)]);
disp(['MAPE',num2str(MAPE)]);
toc
