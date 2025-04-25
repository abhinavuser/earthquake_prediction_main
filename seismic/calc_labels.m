function [Actual_Labels] = Calc_Actual_Labels(Cat,Events_n,Horizon)
    %This Code calculates the Actual Earthquake Magnitude Occured During
    %"Horizon" Period, after the Final Date of Event used for calculation of
    %Corresponiding Parameter

%     clc; clear all;
%     Catalog=xlsread('Region_MN_1976-2013_Hindukush_Polygon_4_above');
%     Events_n=50;
%     Horizon=7;
    Catalog=xlsread(Cat);
    Actual_Labels=[];
    Days=Catalog(:,1:3);
    Mag=Catalog(:,10);

    %     SD=strcat(num2str(Days(1,2)),'/',num2str(Days(1,3)),'/',num2str(Days(1,1)));
    %     ED=strcat(num2str(Days(60,2)),'/',num2str(Days(60,3)),'/',num2str(Days(60,1)));
    %     T_Days= daysdif(SD,ED);
%(Events_n+1)
    for i= (Events_n+1):size(Mag,1)
       % i
            Date_Parameters=Days(i,:);
            %When i is at Last Index, Inster 0 as Actual Label
            if i==size(Mag,1)
                Actual_Mag=0;
                break
            end
            SD=strcat(num2str(Days(i,2)),'/',num2str(Days(i,3)),'/',num2str(Days(i,1)));   

            
            Counter=1;
                      %Check whether next earthquake is "Horizon" days away or
                      %not??
                      %If next earthquake is already horizon days aways,
                      %mark labels as Zero, otherwise search for all the
                      %occured earthquakes and find which one is maximum
            ED=strcat(num2str(Days((i+Counter),2)),'/',num2str(Days((i+Counter),3)),'/',num2str(Days((i+Counter),1)));
            T_Days= daysdif(SD,ED);
            if T_Days > (Horizon+1)  %+1 is added here, because, if there are more events available for the same, they can be traversed via counter update
                Magnitudes=0;
            else
                while(T_Days < (Horizon+1))
                    Counter=Counter+1;
                        %When the "i" is in last few indices and COunter exceeds
                        %the index of catalog, so use break to exit catalog and
                        %deal with the reamainng available events. 
                        if (i+Counter)>size(Mag,1)
                            break
                        end
                    ED=strcat(num2str(Days((i+Counter),2)),'/',num2str(Days((i+Counter),3)),'/',num2str(Days((i+Counter),1)));
                    T_Days= daysdif(SD,ED);
                end
                Counter=Counter-1;
                Magnitudes=Mag((i+1):(i+Counter));
            end


            Actual_Mag=max(Magnitudes);
            
            if isempty(Actual_Mag)     %This case occurs when NExt event is "Horizon+1" Days away
                Actual_Mag=0;
            end
            
            Actual_Labels= [Actual_Labels; Date_Parameters Actual_Mag];
    end
    Actual_Labels= [Actual_Labels; Date_Parameters Actual_Mag];
        
            
            
            
    
    
