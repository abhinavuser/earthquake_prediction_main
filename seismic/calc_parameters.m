function [Parameters] = Calculate_Parameters(Cat,Events_n,Horizon)
    %This code is computing Parameters for every "Events_n" in the catalog. The
    %value of label/target for "Horizon" days will be calculated later in next step/code.
    % Catalog=xlsread('EM_Catalog_2011 onward');
    % Events_n=50;
    Catalog=xlsread(Cat);
    Parameters = [];
    
    M_Threshold = 2.5:0.1:4.5;  %The Thresholding Magnitude for Prediction to be used in calculation of Recurrance Time
    %Horizon=7;
    %Date, Month and Years
    Days=Catalog(:,1:3);
    % SD=strcat(num2str(Days(10,2)),'/',num2str(Days(10,3)),'/',num2str(Days(10,1)));
    % ED=strcat(num2str(Days(60,2)),'/',num2str(Days(60,3)),'/',num2str(Days(60,1)));
    % T_Days= daysdif(SD,ED);

    Mag=Catalog(:,10);

    %Total Duration of Catalog and Total Events in the Catalot
    Total_Events= size(Mag,1);
    First_Date=strcat(num2str(Days(1,2)),'/',num2str(Days(1,3)),'/',num2str(Days(1,1)));
    Last_Date=strcat(num2str(Days(Total_Events,2)),'/',num2str(Days(Total_Events,3)),'/',num2str(Days(Total_Events,1)));
    Total_Duration= daysdif(First_Date,Last_Date);

    %Total Distribution of events Date-wise and then Zero padding for missing
    %days

%
    for i= (Events_n+1):size(Mag,1)
        i
        %now here we select every "Events_n" magnitudes out of all magnitudes
        %and calculate parameters. 

        %Date before which the Parameters are calculated.
        Date_Parameters=Days(i,:);

        %Duration of Days during which "Events_n" Occured
        End_T=strcat(num2str(Days(i,2)),'/',num2str(Days(i,3)),'/',num2str(Days(i,1)));
        Start_T=strcat(num2str(Days(i-Events_n+1,2)),'/',num2str(Days(i-Events_n+1,3)),'/',num2str(Days(i-Events_n+1,1)));
        T_Days= daysdif(Start_T,End_T);

        Mag_n=Mag(i-Events_n+1:i,1);

        %--a & b Value calculation using both maximum liklihood and Least Square
        %method
        [bvalue_mlik, avalue_mlik, std_err] = bval_maxlkh2(Mag_n);
        [bvalue_lsq, avalue_lsq, std_err]   = bval_lsqreg(Mag_n,0.1);

        %--Standard Deviation of Both B Values 
        S =(sum((Mag_n-mean(Mag_n)).^2))/(Events_n*(Events_n-1));
        Std_b_mlik = 2.3*bvalue_mlik^2*sqrt(S);
        Std_b_lsq = 2.3*bvalue_lsq^2*sqrt(S);

        %--Total Recurrance Time Calculation
           %T_rec is calculated using both Maximum Liklihood and Least Mean
           %Square
           %Total recurrance time is calculated for all events abov certain
           %threshold Magnitude....
        T_rec_mlik = T_Days./ (10.^(avalue_mlik-bvalue_mlik.*M_Threshold));
        T_rec_lsq = T_Days./ (10.^(avalue_lsq-bvalue_lsq.*M_Threshold));

        %--Calculate Beta Value
           %Calculate using Inter[retation # 1. 
        M=Events_n;
        delta=T_Days/Total_Duration;
        Beta = (M-(Total_Events*delta))/(sqrt(Total_Events*delta*(1-delta))); 

        %--Calculate Z- Value
        R1= (Total_Events-Events_n)/(Total_Duration-T_Days);
        R2= Events_n/T_Days;
              %Find Distribution of the events for "interval time" and rest of the
              %time"
            Days1=Days([1:i-Events_n  (i+1):Total_Events],:); 
            Distrubution1= Event_Distribution(Days1);

            Days2=Days(i-Events_n+1:i,:);
            Distrubution2= Event_Distribution(Days2);
                %So now we have the days in which events came.. Now we perform zero
                %padding to cover rest of the days.
            Distribution_Padded1 = [Distrubution1(:,4) ; zeros(Total_Duration-T_Days-size(Distrubution1,1),1)]; 
            Distribution_Padded2 = [Distrubution2(:,4) ; zeros(T_Days-size(Distrubution2,1),1)];
            %Variance of both intervals
        S1 = var(Distribution_Padded1);
        S2 = var(Distribution_Padded2);

        Z = (R1-R2)/sqrt((S1/(Total_Events-Events_n)) + (S2/Events_n));
        
    %% ----- Pannakat's Parameters  
       % 1--- Mean Magnitude
        Mean_Mag = mean(Mag_n);    
        
       % 2---- Energy Release
         E=10.^(11.8+1.5.*Mag_n);
         E=sqrt(E);
         Energy=sum(E)/T_Days;
         
       % 3----Magnitude Deficit Maximum Liklihood
         M_exp = avalue_mlik/bvalue_mlik;              %   bvalue_mlik, avalue_mlik
         M_obs = max(Mag_n);
         M_def_Lik= M_obs-M_exp;
         
       % 4---Magnitude Deficit LSQ
         M_exp = avalue_lsq/bvalue_lsq;                %   bvalue_lsq, avalue_lsq
         M_obs = max(Mag_n);
         M_def_lsq = M_obs-M_exp;
         
       % ---Summation of mean square deviation from regression line
    s1=0; index=1; N1=[]; N=[];  %Commulative Magnitude
        for j=max(Mag_n):-0.1: min(Mag_n)        %min(Mag_n)  % 4.0
            N1= find(abs(Mag_n-j)<1E-10);
            s1=s1+size(N1,1);
            N(index,:)=[j size(N1,1) s1];
            index=index+1;
        end
        NU_lsq=[]; NU_mlik=[];
                %5 - Mean Square deviation using a,b max_likli
            for j=1:size(N,1)
                N(j,4)= (log10(N(j,3))-(avalue_mlik+(bvalue_mlik*N(j,1))))^2;     % bvalue_mlik, avalue_mlik
            end
            NU_mlik=sum(N(:,4))/(size(Mag_n,1)-1);

                %6 - Mean Square deviation using a,b lsq
            for j=1:size(N,1)
                N(j,5)= (log10(N(j,3))-(avalue_lsq+(bvalue_lsq*N(j,1))))^2;       % bvalue_lsq, avalue_lsq
            end
            NU_lsq=sum(N(:,5))/(size(Mag_n,1)-1);
            
%% ----- Paco's Paramters x6 and x7
         x7_lsq = 10^(-3*bvalue_lsq);
         x7_mlk = 10^(-3*bvalue_mlik);   
       %--------- x6--- Maximum event occurred in previous 7 days excluding
       %current day.
        Diff_days=0; d=0;
        while(1)
        Current_T=strcat(num2str(Days(i,2)),'/',num2str(Days(i,3)),'/',num2str(Days(i,1)));
        Prev_Day=strcat(num2str(Days(i-d,2)),'/',num2str(Days(i-d,3)),'/',num2str(Days(i-d,1)));
        Diff_d=Diff_days;
        Diff_days= daysdif(Prev_Day,Current_T);
        if Diff_days>7
            d=d-1;
            break;
        end
        d=d+1;
        end
        if Diff_d == 0
            x6 = 0;
        else
            Mm = Mag(i-d:i-1);
            x6 = max(Mm);
        end
        
    %% ----- Write all the parameters in one matrix form...
        Parameters = [Parameters;Date_Parameters T_Days bvalue_mlik avalue_mlik bvalue_lsq avalue_lsq Std_b_mlik Std_b_lsq T_rec_mlik T_rec_lsq Beta Z Mean_Mag Energy M_def_Lik M_def_lsq NU_mlik NU_lsq x6 x7_mlk x7_lsq]; 

    end

  % ---------------------------------------------
  % In the end calculate Actual Labels, using function "Calc_Actual_Labels"
  % and put them next to Parameters. 
  %Calculate Labels for a Particular Horizon, The other Parameter, like
  %nM=50 or 100 and "Catalog" will be same for Parameters Calculation
    [Actual_Labels] = Calc_Actual_Labels(Cat,Events_n,Horizon);
   %Combining Corresponding Parameters and Labels
    Parameters = [Parameters Actual_Labels(:,4)];
    %------------------
   %---Saving
   %save('EM,nM=50,dtp=15','Parameters'); %xlswrite('TIPAGE,nM=50,dtp=7',Parameters)
   
    
    