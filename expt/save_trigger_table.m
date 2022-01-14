function save_trigger_table()
    addpath('./toolbox');
    load_trigger_constants;
    TriggerID = 1:numel(triggerNames);
    TriggerName = {};
    for i=TriggerID
        TriggerName{i} = triggerID2Name(i);
    end
    TriggerID = transpose(TriggerID);
    TriggerName = transpose(TriggerName);
    T = table(TriggerID, TriggerName);
    writetable(T, 'trigger_table.csv');