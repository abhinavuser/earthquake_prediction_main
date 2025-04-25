function [Distribution] = Event_Distribution(Days)

    % This code just calculates the date-wise distribution of events in the catalog. 
    % After getting distribution of events, for rest of the days, zero padding
    % is used to complete the data. 
    %This distribution calculaion purpose is just to calculate the "Variance" of
    %the events. Used in calculation of Z- value.

    % % ---Inputs = 1-Magnitudes Vector, 2- Days Vector or Days 3- T_Days for
    % n_events
    Distribution=[];
      counter=0;
      loop_count=0;
      comp=Days(1,3);


    while(loop_count<size(Days,1))
            loop_count=loop_count+1;
        if comp==Days(loop_count,3);
            counter=counter+1;
        else
            Distribution=[Distribution; Days(loop_count-1,:) counter];
            counter=0;
            comp=Days(loop_count,3);
            loop_count=loop_count-1;
        end
    end
    Distribution=[Distribution; Days(loop_count-1,:) counter];

    