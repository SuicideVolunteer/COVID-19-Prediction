function [O Error w q w2] = SFSSMS_func(input_train, target_train,parameters)

%[ input_train, target_train, input_test, target_test, denNumber ] = divideDataset( F_index, divide_rate); % divide the dataset into two subpopulation

[I,J]=size(input_train);
k=parameters(1);
qs=parameters(2);
M=parameters(3); 
popsize = parameters(4);
Max_Gen = parameters(5);
O = zeros(J,Max_Gen);
w = zeros(I,M);
q = zeros(I,M);
w2 = 0;
dim=2*M*size(input_train,1)+1;
down=-5;  up=5;

%paremeters
NoPob=popsize;                % Number of poblation
N_IterTotal=Max_Gen;          % Maximum iteration

phase   = 1;                  % gas phase
beta    = [0.9, 0.5, 0.1];    % movement
alpha   = [0.3, 0.05, 0];     % colides
H       = [0.9, 0.2,  0];     % random
phases  = [0.5, 0.1,-0.1];    % percent of phases
param   = [0.85 0.35 0.1];    % adjust Param

% Random initial solutions
pob = initialization(dim,NoPob,up,down);
dir = rand(NoPob,dim)*2-1;

% Eval Fitness
fitness = evolution_fitness( input_train, target_train, pob, M,k,qs);

% Get Best
 [bestSol, bestFit] = getBest(pob,fitness, zeros(1,dim), 100000000);
 SMS_Convergence=zeros(1,N_IterTotal);

%BA setting
Mp=3;
Ba=ScaleFree(NoPob,Mp);

for ite = 1:N_IterTotal
    
    % movement
    best = repmat(bestSol, NoPob, 1);
    b = sqrt(sum((best-pob+eps).^2,2));
    b = repmat(b,1,dim);
    a = (best-pob)./b;
    dir = dir * (1 - ite/N_IterTotal)*0.5 + a;
    
    % colis
    r = 1 * alpha(phase);
    for i = 1:NoPob - 1
        for j = i+1:NoPob
            rr = norm(pob(i,:) - pob(j,:));
            if rr < r
                c = dir(i,:);
                d = dir(j,:);
                dir(i,:)=d;
                dir(j,:)=c;
            end
        end
    end
    
    v = 1 * beta(phase) * dir;
    pob = pob + v * rand * param(phase);
    
    % random
    for i=1:NoPob
        if rand< H(phase)
            j = fix(rand*dim)+1;
            pob(i,j)= rand;
        end
    end    
    
    % change of phase
    if 1 - ite/N_IterTotal < phases(phase)
        phase = phase + 1;
    end
    % Eval Fitness
    fitness = evolution_fitness( input_train, target_train, pob, M,k,qs);
    
    % update by scale-free networks (BA algorithm)
    VarSize=[1 dim];
    [~,Frank]=sort(fitness);%给fitness排名
    cpob=pob;%复制一个SMS执行后的种群
    for i=1:NoPob
        S1 = ceil(rand(1)*Ba(i).D);%随机取邻居序号
        S2=Ba(i).Sec(S1);%取得邻居排名
        phi=unifrnd(0,1,VarSize);
        pob(Frank(i),:)=cpob(Frank(i),:)+phi.*(cpob(Frank(S2),:)-cpob(Frank(i),:));%更新排名第i的个体的权值向量
    end
    
    % Eval Fitness
    fitness = evolution_fitness( input_train, target_train, pob, M,k,qs);
    % Get Best
    [bestSol, bestFit] = getBest(pob,fitness, bestSol, bestFit);
    
    SMS_Convergence(ite)=min(bestFit);
    display(['The current best optimal value is : ', num2str(bestFit)]);
end
   [~,O] = evolution_fitness( input_train, target_train,bestSol, M,k,qs);
for m=1:M
    w(:,m) =bestSol((1+2*I*(m-1)):(I+2*I*(m-1)))';
    q(:,m) =bestSol((1+I+2*I*(m-1)):(2*m*I))';
    w2=bestSol(2*m*I+1);
end
    Error = SMS_Convergence;
end
function [bestSol, bestFit] = getBest(pob, fitness, bestSol, bestFit)
[~, p] = min(fitness);
if fitness(p) < bestFit
    bestFit = fitness(p);
    bestSol = pob(p,:);
end
end


