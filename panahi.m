 %%% HW5 - BSS - Erfan Panahi 810198369
clc
clear
fprintf("HW#12 - BSS - Erfan Panahi 810198369\n");

%% Problem 1.
T = 1000;
M = 3;
N = 6;
%--- D
while (1)
    D = randn(M,N);
    D = D ./ sqrt(sum(D.^2));
    DDT = (D'*D) - (D'*D) .* eye(6);
    u_D = max(max(abs(DDT)));
    if (u_D < 0.9) 
        break 
    end
end
%--- S
I = randi([1,N],[1,T]);
v = unifrnd(-5,5,1,T);
S = zeros(N,T);
for i = 1:T
    S(I(i),i) = v(i);
end
%--- noise
Noise = 0.1*randn(3,T);
%--- Observations 
X = D*S + Noise;

%% Part a.
scatter3(X(1,:),X(2,:),X(3,:));
title('scatter plot of observations','Interpreter','latex');
xlabel('$X_1$','Interpreter','latex');
ylabel('$X_2$','Interpreter','latex');
zlabel('$X_3$','Interpreter','latex');

%% Part b. MOD
N0 = 1;
Dhat = randn(M,N);
Dhat = Dhat ./ sqrt(sum(Dhat.^2));
for i = 1:500
% D is fixed
    shat = zeros(N,T); 
    for j = 1:T
        shat(:,j) = MP(X(:,j),Dhat,N);
    end
% S is fixed
    Dhat = X * pinv(shat);
    Dhat = Dhat ./ sqrt(sum(Dhat.^2));
end
corr = Dhat' * D;
SPR = sum(sum(abs(corr) >= 0.99) >= 1) / N;
fprintf("MOD : Successful Recovery Rate: %.3f \n",SPR);

%% Error (MOD)
Order = [1,4,6,2,3,5];
Neg = [2,3,5];
Shat_MOD = shat(Order,:);
Shat_MOD(Neg,:) = -Shat_MOD(Neg,:);
E_MOD = trace((S - Shat_MOD)*(S - Shat_MOD)')/trace(S*S');

%% Part c. K-SVD
N0 = 1;
Dhat = randn(M,N);
Dhat = Dhat ./ sqrt(sum(Dhat.^2));
for i = 1:500
% D is fixed
    shat = zeros(N,T); 
    for j = 1:T
        shat(:,j) = MP(X(:,j),Dhat,N);
    end
% S is fixed
    for j = 1:N
        n = 1:N;
        n(j) = [];
        Xr = X - Dhat(:,n) * shat(n,:);
        k = find(shat(j,:) ~= 0);
        mXr = Xr(:,k);
        [U,G,V] = svd(mXr);
        [m,n] = size(G);
        L = min(m,n);
        [~,index] = sort(diag(G(1:L,1:L)));
        Dhat(:,j) = U(:,index(end));
        Dhat = Dhat ./ sqrt(sum(Dhat.^2));
        shat(j,k) = G(index(end),index(end)) * (V(:,index(end)))';
    end
end
corr = Dhat' * D;
SPR = sum(sum(abs(corr) >= 0.99) >= 1) / N;
fprintf("K-SVD : Successful Recovery Rate: %.3f \n",SPR);

%% Error (K-SVD) 
Order = [3,2,6,5,1,4];
Neg = [1];
Shat_KSVD = shat(Order,:);
Shat_KSVD(Neg,:) = -Shat_KSVD(Neg,:);
E_KSVD = trace((S - Shat_KSVD)*(S - Shat_KSVD)')/trace(S*S');

%% Part d. Error 
fprintf("\n MOD: E = %.20f\n",E_MOD);
fprintf(" K-SVD: E = %.20f\n",E_KSVD);

%% Problem 2.
clear
load('hw12.mat')
[M,N] = size(D);
T = size(S,2);
N0 = 3; 

%% MOD
Dhat = randn(M,N);
Dhat = Dhat ./ sqrt(sum(Dhat.^2));
EOR_MOD = zeros(1,100);
for i = 1:100
% D is fixed
    shat = zeros(N,T); 
    for j = 1:T
        shat(:,j) = OMP(X(:,j),Dhat,N,N0);
    end
% S is fixed
    Dhat = X * pinv(shat);
    Dhat = Dhat ./ sqrt(sum(Dhat.^2));
    EOR_MOD(i) = trace((X - Dhat*shat)*(X - Dhat*shat)')/trace(X*X');
end
corr = Dhat' * D;
SPR = sum(sum(abs(corr) >= 0.99) >= 1) / N;
fprintf("MOD : Successful Recovery Rate: %.3f \n",SPR);

%% K-SVD
Dhat = randn(M,N);
Dhat = Dhat ./ sqrt(sum(Dhat.^2));
EOR_KSVD = zeros(1,100);
for i = 1:100
% D is fixed
    shat = zeros(N,T); 
    for j = 1:T
        shat(:,j) = OMP(X(:,j),Dhat,N,N0);
    end
% S is fixed
    for j = 1:N
        n = 1:N;
        n(j) = [];
        Xr = X - Dhat(:,n) * shat(n,:);
        k = find(shat(j,:) ~= 0);
        mXr = Xr(:,k);
        [U,G,V] = svd(mXr);
        [m,n] = size(G);
        L = min(m,n);
        [~,index] = sort(diag(G(1:L,1:L)));
        Dhat(:,j) = U(:,index(end));
        Dhat = Dhat ./ sqrt(sum(Dhat.^2));
        shat(j,k) = G(index(end),index(end)) * (V(:,index(end)))';
    end
    EOR_KSVD(i) = trace((X - Dhat*shat)*(X - Dhat*shat)')/trace(X*X');
end
corr = Dhat' * D;
SPR = sum(sum(abs(corr) >= 0.99) >= 1) / N;
fprintf("K-SVD : Successful Recovery Rate: %.3f \n",SPR);

%% Comparing MOD and K-SVD Error of Representation
itr = 1:100;
plot(itr,EOR_MOD,'b',itr,EOR_KSVD,'r');
legend('MOD','K-SVD');
title('Error of Representation','Interpreter','latex');
xlabel('itr','Interpreter','latex');
ylabel('Error','Interpreter','latex');

%% Functions
function [sMP] = MP(x,D,N)
    % N0 = 1;
    sMP = zeros(N,1);
    ro = x'*D;
    [~,posMP] = max(abs(ro));
    sMP(posMP) = ro(posMP);
end

function [sOMP] = OMP(x,D,N,N0)
    x1 = x;
    posOMP = zeros(1,N0);
    sOMP = zeros(N,1);
    tic
    for i=1:N0
       ro = x1' * D;
       [~,posOMP(i)] = max(abs(ro));
       if i>1
           Dsub=D(:,posOMP(1:i));
           sOMP(posOMP(1:i)) = pinv(Dsub) * x;
           x1 = x - D * sOMP;
       else
           sOMP(posOMP(1)) = ro(posOMP(1));
           x1 = x1 - sOMP(posOMP(1)) * D(:,posOMP(1)); 
       end
    end
end



    
