function [best_fitness, MMRE_best, PRED_25_best, Med_Ab_Res_best,...
         Sum_Ab_Res_best, SD_Ab_Res_best, best_rf] = ...
          PSO(population_size, number_of_iterations, regression_method,...
          X_train, Y_train, X_test, Y_test, dataset_name)


if strcmp(regression_method, 'SVR with RBF kernel')
    parameters_number = 3;
    range_of_parmaters{1} = [80, 150];   %--> C
    range_of_parmaters{2} = [0.01, 1]; %--> epsilon
    range_of_parmaters{3} = [0.01, 1]; %--> gamma

elseif strcmp(regression_method, 'M5P')
    parameters_number = 3;
    range_of_parmaters{1} = [2, 20];   %--> I
    range_of_parmaters{2} = [0, 1]; %--> P
    range_of_parmaters{3} = [0, 1]; %--> S
end

w=0.2;              % Inertia Weight
wdamp=1;            % Inertia Weight Damping Ratio
c1=0.7;             % Personal Learning Coefficient
c2=1.0;             % Global Learning Coefficient


number_of_features = size(X_train, 2);
dimension = 2*(parameters_number) + number_of_features;

%%%%% initialize population:
particles = zeros(population_size, dimension);
for particle_index = 1:population_size
    while true
        particles(particle_index, :) = round(rand(1, dimension));
        selected_features = particles(particle_index,...
            end-number_of_features+1 : end);
        if ~sum(selected_features) == 0  %--> not all features are removed
            break
        end
    end
end
best_parameters_personal= zeros(population_size,parameters_number);
for parameter_index = 1:parameters_number
    VarMin = range_of_parmaters{parameter_index}(1);
    VarMax = range_of_parmaters{parameter_index}(2);
    VelMax = 0.1*(VarMax-VarMin);
    VelMin = -VelMax;
    particles(:,2*(parameter_index-1) + 1) =...
        VarMin+(VarMax-VarMin)*rand(population_size,1);  
    best_parameters_personal(:,parameter_index)=...
        particles(:,2*(parameter_index-1) + 1);
    particles(:,2*(parameter_index-1) + 2) =...
        VelMin+(VelMax-VelMin)*rand(population_size,1);   
    
end


%% PSO Main Loop


best_fitness_personal= Inf(population_size,1);
best_fitness= Inf; 
best_parameters_global= best_parameters_personal(1,:);
for iteration = 1:number_of_iterations
    
    fitness = zeros(population_size, 1); 
    MMRE = zeros(population_size, 1);
    PRED_25 = zeros(population_size, 1); 
    Sum_Ab_Res = zeros(population_size, 1);
    Med_Ab_Res = zeros(population_size, 1); 
    SD_Ab_Res = zeros(population_size, 1);
    for particle_index = 1:population_size
        
        particle = particles(particle_index, :);
        %%%%% extract parameters out of particles:
        parameters = zeros(parameters_number, 1);
        velocities = zeros(parameters_number, 1); 
        for parameter_index = 1:parameters_number
            parameters(parameter_index) =...
                particle(2*(parameter_index-1) + 1);
            velocities(parameter_index) =...
                particle(2*(parameter_index-1) + 2);            
        end
        
        
        for parameter_index = 1:parameters_number
            VarMin = range_of_parmaters{parameter_index}(1);
            VarMax = range_of_parmaters{parameter_index}(2);
            VelMax = 0.1*(VarMax-VarMin);
            VelMin = -VelMax;
            r1 = rand;
            r2 = rand;
            velocity = velocities(parameter_index);
            position = parameters(parameter_index);
            % Update Velocity
            velocity = w*velocity ...
                +c1*r1*(best_parameters_personal...
                (particle_index, parameter_index)-position) ...
                +c2*r2*(best_parameters_global(parameter_index)-position);
        
            % Apply Velocity Limits
            velocity = max(velocity,VelMin);
            velocity = min(velocity,VelMax);
        
            % Update Position
            position = position + velocity;
        
            % Velocity Mirror Effect
            IsOutside=(position<VarMin | position>VarMax);
            velocity(IsOutside)=-velocity(IsOutside);
        
            % Apply Position Limits
            position = max(position,VarMin);
            position = min(position,VarMax);
            
          %  disp(position);
          %  disp(velocity);
            
            particle(2*(parameter_index-1) + 1) = position;
            particle(2*(parameter_index-1) + 2) = velocity;
        
        end
        particles(particle_index, :) = particle;
        %disp(['The result is: [' num2str(parameters(:).') ']']) ;
        %%%%% extract selected features out of chromosome:
        
        selected_features = particle(end-number_of_features+1 : end);
        if sum(selected_features) == 0
            a_feature_to_be_mutated =...
                round(1 + (number_of_features - 1) * rand);
            particles(particle_index,...
                end-number_of_features+a_feature_to_be_mutated) = 1;
            selected_features =...
                particles(particle_index, end-number_of_features+1 : end);
        end
        
        %%%%% evaluate the fitness of particles using PRED(25) and MMRE:
        [fitness(particle_index), MMRE(particle_index),...
            PRED_25(particle_index), Sum_Ab_Res(particle_index),...
            Med_Ab_Res(particle_index), SD_Ab_Res(particle_index)] = ...
        calculate_fitness(X_train, Y_train, X_test, Y_test,...
        regression_method, parameters, selected_features, dataset_name);
            
        % Update Personal Best
        if (fitness(particle_index)< best_fitness_personal(particle_index))
            best_fitness_personal(particle_index)= fitness(particle_index);
           % disp(best_fitness_personal);
            best_particle_index_personal = particle_index;
            best_parameters_personal(particle_index,:) = parameters;
            if best_fitness_personal(particle_index) < best_fitness
                best_fitness = best_fitness_personal(particle_index);
                best_particle_index = particle_index;
                best_parameters_global = parameters;
                best_particle = particles(best_particle_index, :);
                best_selected_features =...
                    best_particle(end-number_of_features+1 : end);
                best_rf = sum(best_selected_features == 0);
                MMRE_best = MMRE(best_particle_index);
                PRED_25_best = PRED_25(best_particle_index);
                Sum_Ab_Res_best = Sum_Ab_Res(best_particle_index);
                Med_Ab_Res_best = Med_Ab_Res(best_particle_index);
                SD_Ab_Res_best = SD_Ab_Res(best_particle_index);
            end
        end
        
        
    end
    

    
    w=w*wdamp;
    
end