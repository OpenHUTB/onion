
% Analyzing behavioral data
% Four button responses were logged in maximum

if ~exist('SampleData')
   error('Error. \nNo SampleData Folder.')    
end
SaveDir = [pwd '/SampleResult/'];
DataDir = [pwd '/SampleData/'];

tAcc = [];
for ss = 1:6
    ID = ['sub-0' num2str(ss)];

    Tasks=[];
    TFtotal =[];
    for Run = 1:18
        
        % Load log file
        filename = [DataDir 'BehavData/' ID '/Behav_' ID '_' num2str(Run) '_Log.txt'];
        fileID = fopen(filename);
        data = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s %s %s','Delimiter','\t');

        % TaskType
        Type =  str2num(char(data{2}));

        % Correct Button
        Correct = str2num(char(data{5}));

        % First Answer
        Ans1 = str2num(char(data{6}));
        RT1 =  str2num(char(data{7}));

        % Second Answer
        Ans2 = str2num(char(data{8}));
        RT2 =  str2num(char(data{9}));

        % Third Answer
        Ans3 = str2num(char(data{10}));
        RT3 =  str2num(char(data{11}));

        % Fourth Answer
        Ans4 = str2num(char(data{12}));
        RT4 =  str2num(char(data{13}));


        %Trial with True/False 
        Ind = find(Correct);
        T = Type(Ind);
        C = Correct(Ind);
        A1 = Ans1(Ind);
        R1 = RT1(Ind);
        A2 = Ans2(Ind);
        R2 = RT2(Ind);  
        A3 = Ans3(Ind);
        R3 = RT3(Ind);  
        A4 = Ans4(Ind);
        R4 = RT4(Ind);  

        TF = [];

        for ii = 1:length(C)
            % Exclude too early (< 1000ms) button responses 
            if R1(ii)> 1000
                if C(ii) == A1(ii)
                    TF(ii)=1;
                else
                    TF(ii)=0;
                end	
            elseif R2(ii)>1000
                if C(ii) == A2(ii)
                    TF(ii)=1;
                else
                    TF(ii)=0;
                end		
            elseif R3(ii) > 1000
                if C(ii) == A3(ii)
                    TF(ii)=1;
                else
                    TF(ii)=0;
                end	 		
            elseif R4(ii)> 1000
                if C(ii) == A4(ii)
                    TF(ii)=1;
                else
                    TF(ii)=0;
                end	
            else
                TF(ii)=0;
            end
        end
        Tasks = [Tasks; T];
        TFtotal = [TFtotal; TF'];

    end

    Z = zeros(105,1);

    for kk = 1: length(Tasks)
        Z(Tasks(kk))=Z(Tasks(kk))+TFtotal(kk);
    end

    TaskType = find(Z);
    Z= 100*Z(find(Z))/12;

    tAcc(:,ss) = Z;

end

Fig = figure;
boxplot(tAcc)
xlabel('Subjects')
ylabel('Task performance (%)')
saveas(Fig, [SaveDir 'AllSubPerformance.eps'])
save([SaveDir 'AllSubPerformance.mat'], 'tAcc');
