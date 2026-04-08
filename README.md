# NOA
%% 1. Map
xRange = [0, 100]; 
yRange = [0, 100]; 
zRange = [0, 100];  
startPoint = [10, 10, 15];  
endPoint = [90, 90, 25];   

obstacles = [ 40, 40, 25, 8;   
              60, 40, 20, 10; 
              40, 70, 30, 12;
              75, 70, 18, 7;
              65, 65, 22, 9;   
              30, 60, 28, 8]; 

softObstacles = [ 30, 50, 22, 5; 
                  55, 60, 15, 4; 
                  80, 30, 28, 6;
                  45, 45, 24, 5;  
                  55, 40, 18, 6];

[xGrid, yGrid] = meshgrid(linspace(xRange(1),xRange(2),50), linspace(yRange(1),yRange(2),50));
zTerrain = 5 + 3*sin(xGrid/10) + 2*cos(yGrid/15) + randn(size(xGrid))*0.8;
zTerrain(zTerrain < 0) = 0;

color_PSO = [0, 0.4470, 0.7410];    
color_GA  = [0.9290, 0.6940, 0.1250];
color_GWO = [0.4940, 0.1840, 0.5560];
color_SSA = [0.4660, 0.6740, 0.1880];
color_seabed = [0.8, 0.9, 0.95]; 
color_hard_obs = [0.3, 0.3, 0.3];  
color_soft_obs = [0.8, 0.85, 0.9];  
edge_soft_obs = [0.5, 0.6, 0.8];    

popSize = 60;          
maxIter = 150;         
dim = 3;               
lb = [xRange(1), yRange(1), zRange(1)+0.5];  
ub = [xRange(2), yRange(2), zRange(2)];      
pathNum = 11;         
updateInterval = 2;   

%% 2. Algorithm

pso_w = 0.8;           
pso_c1 = 2;           
pso_c2 = 2;            
pso_velMax = (ub - lb) * 0.2;  
pso_velMin = -pso_velMax;       

ga_crossoverProb = 0.8; 
ga_mutationProb = 0.15; 
ga_tournamentSize = 3;  
ga_mutationDecay = 0.98;

gwo_a_start = 2;       
gwo_a_end = 0;       
gwo_A_max = 2;        
gwo_C_max = 2;        

ssa_pd = 0.2;          
ssa_sd = 0.1;          
ssa_ST = 0.8;        
ssa_A = 1.5;           
ssa_fl = 0.5;          
ssa_P = 0.7;           

%% 3. Initialize
[pso_pop, pso_vel, pso_pbest, pso_pbestFitness, pso_gbest, pso_gbestFitness] = initPSO(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain, pso_velMin, pso_velMax);

[ga_pop, ~, ga_globalBestPath] = initGA(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain);

[gwo_pop, gwo_globalBestFitness, gwo_globalBestPath] = initGWO(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain);

[ssa_pop, ssa_globalBestFitness, ssa_globalBestPath] = initSSA(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain);

%% 4. Fitness
fitness = @(path) calculateFitnessAUV(path, startPoint, endPoint, obstacles, softObstacles, xGrid, yGrid, zTerrain);

%% 5. Dynamic Path Comparison
figure('Position', [50, 50, 1200, 800]);  
hold on; grid on; axis equal;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Dynamic Path Comparison（Iteration0/%d）', maxIter));  

xlim([0, 100]);
ylim([0, 100]);
zlim([0, 100]);

drawEnvironment(xRange, yRange, obstacles, softObstacles, startPoint, endPoint, color_seabed, color_hard_obs, color_soft_obs, edge_soft_obs, xGrid, yGrid, zTerrain);

pso_path_hdl = plot3([], [], [], 'Color', color_PSO, 'LineWidth', 3);  
ga_path_hdl  = plot3([], [], [], 'Color', color_GA,  'LineWidth', 3); 
gwo_path_hdl = plot3([], [], [], 'Color', color_GWO, 'LineWidth', 3); 
ssa_path_hdl = plot3([], [], [], 'Color', color_SSA, 'LineWidth', 3); 

pso_node_hdl = scatter3([], [], [], 50, color_PSO, 'filled', 'MarkerEdgeColor', 'k');  
ga_node_hdl  = scatter3([], [], [], 50, color_GA,  'filled', 'MarkerEdgeColor', 'k');  
gwo_node_hdl = scatter3([], [], [], 50, color_GWO, 'filled', 'MarkerEdgeColor', 'k'); 
ssa_node_hdl = scatter3([], [], [], 50, color_SSA, 'filled', 'MarkerEdgeColor', 'k'); 

legend_created = false;

%% 6. Interation And Update
pso_bestFitness = zeros(maxIter, 1);  
ga_bestFitness  = zeros(maxIter, 1);  
gwo_bestFitness = zeros(maxIter, 1);  
ssa_bestFitness = zeros(maxIter, 1);  
pso_fit = zeros(popSize, 1);
ga_fit  = zeros(popSize, 1);
gwo_fit = zeros(popSize, 1);
ssa_fit = zeros(popSize, 1);  

for i = 1:popSize
    pso_fit(i) = fitness(pso_pop(i,:,:));
    ga_fit(i)  = fitness(ga_pop(i,:,:));
    gwo_fit(i) = fitness(gwo_pop(i,:,:));
    ssa_fit(i) = fitness(ssa_pop(i,:,:));
end

[currentGABestFit, currentGABestIdx] = min(ga_fit);
ga_globalBestFitness = currentGABestFit;
ga_globalBestPath = ga_pop(currentGABestIdx,:,:);

[currentGWOBestFit, currentGWOBestIdx] = min(gwo_fit);
gwo_globalBestFitness = currentGWOBestFit;
gwo_globalBestPath = gwo_pop(currentGWOBestIdx,:,:);

[currentSSABestFit, currentSSABestIdx] = min(ssa_fit);
ssa_globalBestFitness = currentSSABestFit;
ssa_globalBestPath = ssa_pop(currentSSABestIdx,:,:);

for iter = 1:maxIter

    w = pso_w;
    for i = 1:popSize
        for j = 1:pathNum
            r1 = reshape(rand(1,dim), 1, 1, 3);
            r2 = reshape(rand(1,dim), 1, 1, 3);
            gbest_node = pso_gbest(1,j,:);
            pbest_node = pso_pbest(i,j,:);
            pop_node = pso_pop(i,j,:);
            pso_vel(i,j,:) = w * pso_vel(i,j,:) + pso_c1*r1.*(pbest_node - pop_node) + pso_c2*r2.*(gbest_node - pop_node);
             pso_vel(i,j,1) = max(min(pso_vel(i,j,1), pso_velMax(1)), pso_velMin(1));
            pso_vel(i,j,2) = max(min(pso_vel(i,j,2), pso_velMax(2)), pso_velMin(2));
            pso_vel(i,j,3) = max(min(pso_vel(i,j,3), pso_velMax(3)), pso_velMin(3));
            pso_pop(i,j,:) = pop_node + pso_vel(i,j,:);
        end
        pso_pop(i,:,:) = smoothPathNodes(pso_pop(i,:,:), 2);
        pso_pop(i,:,1) = max(min(pso_pop(i,:,1), ub(1)), lb(1));
        pso_pop(i,:,2) = max(min(pso_pop(i,:,2), ub(2)), lb(2));
        pso_pop(i,:,3) = max(min(pso_pop(i,:,3), ub(3)), lb(3));
        pso_fit(i) = fitness(pso_pop(i,:,:));
        if pso_fit(i) < pso_pbestFitness(i)
            pso_pbestFitness(i) = pso_fit(i);
            pso_pbest(i,:,:) = pso_pop(i,:,:);
        end
    end
    [currentPSOBestFit, currentPSOBestIdx] = min(pso_fit);
    if currentPSOBestFit < pso_gbestFitness
        pso_gbestFitness = currentPSOBestFit;
        pso_gbest = pso_pop(currentPSOBestIdx,:,:);
    end
    pso_bestFitness(iter) = pso_gbestFitness;

    ga_selected = tournamentSelection(ga_pop, ga_fit, ga_tournamentSize);
    ga_crossed = crossover(ga_selected, ga_crossoverProb, lb, ub);
    currentGaMutateProb = ga_mutationProb * (ga_mutationDecay^iter);
    ga_mutated = mutation(ga_crossed, currentGaMutateProb, lb, ub, pathNum, dim);
    for i = 1:popSize
        ga_mutated(i,:,:) = smoothPathNodes(ga_mutated(i,:,:), 2);
        ga_mutated(i,:,1) = max(min(ga_mutated(i,:,1), ub(1)), lb(1));
        ga_mutated(i,:,2) = max(min(ga_mutated(i,:,2), ub(2)), lb(2));
        ga_mutated(i,:,3) = max(min(ga_mutated(i,:,3), ub(3)), lb(3));
    end
    ga_pop = ga_mutated;
    for i = 1:popSize
        ga_fit(i) = fitness(ga_pop(i,:,:));
    end
    [currentGABestFit, currentGABestIdx] = min(ga_fit);
    if currentGABestFit < ga_globalBestFitness
        ga_globalBestFitness = currentGABestFit;
        ga_globalBestPath = ga_pop(currentGABestIdx,:,:);
    end
    ga_bestFitness(iter) = ga_globalBestFitness;

    a = gwo_a_start - (gwo_a_start - gwo_a_end) * iter / maxIter;
    [sortedGWOFit, sortedGWOIdx] = sort(gwo_fit);
    alphaWolf = gwo_pop(sortedGWOIdx(1),:,:);  
    betaWolf  = gwo_pop(sortedGWOIdx(2),:,:);  
    deltaWolf = gwo_pop(sortedGWOIdx(3),:,:);  
    for i = 1:popSize
        for j = 1:pathNum
            A1 = a * (2 * rand(1,dim) - 1);
            A2 = a * (2 * rand(1,dim) - 1);
            A3 = a * (2 * rand(1,dim) - 1);
            C1 = 2 * rand(1,dim);
            C2 = 2 * rand(1,dim);
            C3 = 2 * rand(1,dim);
            D_alpha = abs(C1 .* reshape(alphaWolf(1,j,:),1,3) - reshape(gwo_pop(i,j,:),1,3));
            D_beta  = abs(C2 .* reshape(betaWolf(1,j,:),1,3) - reshape(gwo_pop(i,j,:),1,3));
            D_delta = abs(C3 .* reshape(deltaWolf(1,j,:),1,3) - reshape(gwo_pop(i,j,:),1,3));
            X1 = reshape(alphaWolf(1,j,:),1,3) - A1 .* D_alpha;
            X2 = reshape(betaWolf(1,j,:),1,3) - A2 .* D_beta;
            X3 = reshape(deltaWolf(1,j,:),1,3) - A3 .* D_delta;
            newPos = (X1 + X2 + X3) / 3;
            gwo_pop(i,j,:) = reshape(newPos, 1, 1, 3);
        end
        gwo_pop(i,:,:) = smoothPathNodes(gwo_pop(i,:,:), 2);
        gwo_pop(i,:,1) = max(min(gwo_pop(i,:,1), ub(1)), lb(1));
        gwo_pop(i,:,2) = max(min(gwo_pop(i,:,2), ub(2)), lb(2));
        gwo_pop(i,:,3) = max(min(gwo_pop(i,:,3), ub(3)), lb(3));
    end
    for i = 1:popSize
        gwo_fit(i) = fitness(gwo_pop(i,:,:));
    end
    [currentGWOBestFit, currentGWOBestIdx] = min(gwo_fit);
    if currentGWOBestFit < gwo_globalBestFitness
        gwo_globalBestFitness = currentGWOBestFit;
        gwo_globalBestPath = gwo_pop(currentGWOBestIdx,:,:);
    end
    gwo_bestFitness(iter) = gwo_globalBestFitness;
    [sortedSSAFit, sortedSSAIdx] = sort(ssa_fit);
    producerNum = round(ssa_pd * popSize); 
    scouterNum = round(ssa_sd * popSize);  
    producers = ssa_pop(sortedSSAIdx(1:producerNum), :, :);  
    followers = ssa_pop(sortedSSAIdx(producerNum+1:end-scouterNum), :, :); 
    scouters = ssa_pop(sortedSSAIdx(end-scouterNum+1:end), :, :); 
    for i = 1:producerNum
        for j = 1:pathNum
            if rand() < ssa_ST  
                step_vec = ssa_A * (1 - rand()) * (ub - lb); 
                step = reshape(step_vec, 1, 1, 3);          
                ssa_pop(sortedSSAIdx(i),j,:) = producers(i,j,:) + step .* randn(1,1,3);
            else  
                rand_vec = lb + rand(1,3) .* (ub - lb);     
                randPos = reshape(rand_vec, 1, 1, 3);       
                ssa_pop(sortedSSAIdx(i),j,:) = randPos;
            end
        end
        ssa_pop(sortedSSAIdx(i),:,:) = smoothPathNodes(ssa_pop(sortedSSAIdx(i),:,:), 2);
        ssa_pop(sortedSSAIdx(i),:,1) = max(min(ssa_pop(sortedSSAIdx(i),:,1), ub(1)), lb(1));
        ssa_pop(sortedSSAIdx(i),:,2) = max(min(ssa_pop(sortedSSAIdx(i),:,2), ub(2)), lb(2));
        ssa_pop(sortedSSAIdx(i),:,3) = max(min(ssa_pop(sortedSSAIdx(i),:,3), ub(3)), lb(3));
    end
    followerNum = popSize - producerNum - scouterNum;
    for i = 1:followerNum
        for j = 1:pathNum
            if i > followerNum/2  
                randProducer = randi(producerNum);
                step_fl_vec = ssa_fl * randn(1,3) .* (ub - lb);
                step_fl = reshape(step_fl_vec, 1, 1, 3);
                ssa_pop(sortedSSAIdx(producerNum+i),j,:) = producers(randProducer,j,:) + abs(producers(randProducer,j,:) - ssa_pop(sortedSSAIdx(producerNum+i),j,:)) .* step_fl;
            else  
                step_p_vec = (ub - lb) .* randn(1,3) * ssa_P;
                step_p = reshape(step_p_vec, 1, 1, 3);
                ssa_pop(sortedSSAIdx(producerNum+i),j,:) = producers(1,j,:) + step_p;
            end
        end
        ssa_pop(sortedSSAIdx(producerNum+i),:,:) = smoothPathNodes(ssa_pop(sortedSSAIdx(producerNum+i),:,:), 2);
        ssa_pop(sortedSSAIdx(producerNum+i),:,1) = max(min(ssa_pop(sortedSSAIdx(producerNum+i),:,1), ub(1)), lb(1));
        ssa_pop(sortedSSAIdx(producerNum+i),:,2) = max(min(ssa_pop(sortedSSAIdx(producerNum+i),:,2), ub(2)), lb(2));
        ssa_pop(sortedSSAIdx(producerNum+i),:,3) = max(min(ssa_pop(sortedSSAIdx(producerNum+i),:,3), ub(3)), lb(3));
    end
    for i = 1:scouterNum
        for j = 1:pathNum
            if sortedSSAFit(end-scouterNum+i) > ssa_globalBestFitness  
                step_sc1_vec = 0.05 * randn(1,3) .* (ub - lb);
                step_sc1 = reshape(step_sc1_vec, 1, 1, 3);
                ssa_pop(sortedSSAIdx(end-scouterNum+i),j,:) = ssa_globalBestPath(1,j,:) + step_sc1;
            else  
                randStep_sc_vec = (-1)^round(rand()) * 0.1 * randn(1,3);
                randStep_sc = reshape(randStep_sc_vec .* (ub - lb), 1, 1, 3);
                ssa_pop(sortedSSAIdx(end-scouterNum+i),j,:) = ssa_pop(sortedSSAIdx(end-scouterNum+i),j,:) + randStep_sc;
            end
        end
        ssa_pop(sortedSSAIdx(end-scouterNum+i),:,:) = smoothPathNodes(ssa_pop(sortedSSAIdx(end-scouterNum+i),:,:), 2);
        ssa_pop(sortedSSAIdx(end-scouterNum+i),:,1) = max(min(ssa_pop(sortedSSAIdx(end-scouterNum+i),:,1), ub(1)), lb(1));
        ssa_pop(sortedSSAIdx(end-scouterNum+i),:,2) = max(min(ssa_pop(sortedSSAIdx(end-scouterNum+i),:,2), ub(2)), lb(2));
        ssa_pop(sortedSSAIdx(end-scouterNum+i),:,3) = max(min(ssa_pop(sortedSSAIdx(end-scouterNum+i),:,3), ub(3)), lb(3));
    end
    for i = 1:popSize
        ssa_fit(i) = fitness(ssa_pop(i,:,:));
    end
    [currentSSABestFit, currentSSABestIdx] = min(ssa_fit);
    if currentSSABestFit < ssa_globalBestFitness
        ssa_globalBestFitness = currentSSABestFit;
        ssa_globalBestPath = ssa_pop(currentSSABestIdx,:,:);
    end
    ssa_bestFitness(iter) = ssa_globalBestFitness;

    if mod(iter, updateInterval) == 0 || iter == 1 || iter == maxIter
        pso_bestPath = reshape(pso_gbest, [], 3);
        pso_fullPath = [startPoint; pso_bestPath; endPoint];
        pso_smoothPath = bsplineSmooth(pso_fullPath, 3);
        set(pso_path_hdl, 'XData', pso_smoothPath(:,1), 'YData', pso_smoothPath(:,2), 'ZData', pso_smoothPath(:,3));
        set(pso_node_hdl, 'XData', pso_fullPath(:,1), 'YData', pso_fullPath(:,2), 'ZData', pso_fullPath(:,3));

        ga_bestPath = reshape(ga_globalBestPath, [], 3);
        ga_fullPath = [startPoint; ga_bestPath; endPoint];
        ga_smoothPath = bsplineSmooth(ga_fullPath, 3);
        set(ga_path_hdl, 'XData', ga_smoothPath(:,1), 'YData', ga_smoothPath(:,2), 'ZData', ga_smoothPath(:,3));
        set(ga_node_hdl, 'XData', ga_fullPath(:,1), 'YData', ga_fullPath(:,2), 'ZData', ga_fullPath(:,3));

        gwo_bestPath = reshape(gwo_globalBestPath, [], 3);
        gwo_fullPath = [startPoint; gwo_bestPath; endPoint];
        gwo_smoothPath = bsplineSmooth(gwo_fullPath, 3);
        set(gwo_path_hdl, 'XData', gwo_smoothPath(:,1), 'YData', gwo_smoothPath(:,2), 'ZData', gwo_smoothPath(:,3));
        set(gwo_node_hdl, 'XData', gwo_fullPath(:,1), 'YData', gwo_fullPath(:,2), 'ZData', gwo_fullPath(:,3));
        
        ssa_bestPath = reshape(ssa_globalBestPath, [], 3);
        ssa_fullPath = [startPoint; ssa_bestPath; endPoint];
        ssa_smoothPath = bsplineSmooth(ssa_fullPath, 3);
        set(ssa_path_hdl, 'XData', ssa_smoothPath(:,1), 'YData', ssa_smoothPath(:,2), 'ZData', ssa_smoothPath(:,3));
        set(ssa_node_hdl, 'XData', ssa_fullPath(:,1), 'YData', ssa_fullPath(:,2), 'ZData', ssa_fullPath(:,3));
        
        title(sprintf('Dynamic Path Comparison（Iteration%d/%d）\nPSO：%.2f | GA：%.2f | GWO：%.2f | SSA：%.2f', ...
            iter, maxIter, pso_gbestFitness, ga_globalBestFitness, gwo_globalBestFitness, ssa_globalBestFitness));
        
        if ~legend_created
            legend([pso_path_hdl, ga_path_hdl, gwo_path_hdl, ssa_path_hdl], ...
                'PSO', 'GA', 'GWO', 'SSA', 'Location', 'best');
            legend_created = true;
        end
        drawnow;
     end
end
%% 7. Convergence Curve
figure('Position', [100, 100, 1200, 600]);
hold on; grid on;
xlabel('Iterations'); ylabel('Fitness');
title(' Convergence Curve');
plot(1:maxIter, pso_bestFitness, 'Color', color_PSO, 'LineWidth', 2, 'DisplayName', 'PSO');
plot(1:maxIter, ga_bestFitness,  'Color', color_GA,  'LineWidth', 2, 'DisplayName', 'GA');
plot(1:maxIter, gwo_bestFitness, 'Color', color_GWO, 'LineWidth', 2, 'DisplayName', 'GWO');
plot(1:maxIter, ssa_bestFitness, 'Color', color_SSA, 'LineWidth', 2, 'DisplayName', 'SSA');
legend('Location', 'best');
xlim([1, maxIter]);
allFitness = [pso_bestFitness; ga_bestFitness; gwo_bestFitness; ssa_bestFitness];
ylim([min(allFitness)*0.9, max(allFitness)*1.1]);
grid on; 
box on;

%% 8. Path Comparison
figure('Position', [150, 150, 1200, 800]);
hold on; grid on; axis equal;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Path planning diagram');  

xlim([0, 100]);
ylim([0, 100]);
zlim([0, 100]);
drawEnvironment(xRange, yRange, obstacles, softObstacles, startPoint, endPoint, color_seabed, color_hard_obs, color_soft_obs, edge_soft_obs, xGrid, yGrid, zTerrain);

pso_bestPath = reshape(pso_gbest, [], 3);
pso_fullPath = [startPoint; pso_bestPath; endPoint];
pso_smoothPath = bsplineSmooth(pso_fullPath, 3);
pso_final_hdl = plot3(pso_smoothPath(:,1), pso_smoothPath(:,2), pso_smoothPath(:,3), 'Color', color_PSO, 'LineWidth', 3);
scatter3(pso_fullPath(:,1), pso_fullPath(:,2), pso_fullPath(:,3), 50, color_PSO, 'filled', 'MarkerEdgeColor', 'k');

ga_bestPath = reshape(ga_globalBestPath, [], 3);
ga_fullPath = [startPoint; ga_bestPath; endPoint];
ga_smoothPath = bsplineSmooth(ga_fullPath, 3);
ga_final_hdl = plot3(ga_smoothPath(:,1), ga_smoothPath(:,2), ga_smoothPath(:,3), 'Color', color_GA, 'LineWidth', 3);
scatter3(ga_fullPath(:,1), ga_fullPath(:,2), ga_fullPath(:,3), 50, color_GA, 'filled', 'MarkerEdgeColor', 'k');

gwo_bestPath = reshape(gwo_globalBestPath, [], 3);
gwo_fullPath = [startPoint; gwo_bestPath; endPoint];
gwo_smoothPath = bsplineSmooth(gwo_fullPath, 3);
gwo_final_hdl = plot3(gwo_smoothPath(:,1), gwo_smoothPath(:,2), gwo_smoothPath(:,3), 'Color', color_GWO, 'LineWidth', 3);
scatter3(gwo_fullPath(:,1), gwo_fullPath(:,2), gwo_fullPath(:,3), 50, color_GWO, 'filled', 'MarkerEdgeColor', 'k');

ssa_bestPath = reshape(ssa_globalBestPath, [], 3);
ssa_fullPath = [startPoint; ssa_bestPath; endPoint];
ssa_smoothPath = bsplineSmooth(ssa_fullPath, 3);
ssa_final_hdl = plot3(ssa_smoothPath(:,1), ssa_smoothPath(:,2), ssa_smoothPath(:,3), 'Color', color_SSA, 'LineWidth', 3);
scatter3(ssa_fullPath(:,1), ssa_fullPath(:,2), ssa_fullPath(:,3), 50, color_SSA, 'filled', 'MarkerEdgeColor', 'k');

legend([pso_final_hdl, ga_final_hdl, gwo_final_hdl, ssa_final_hdl], ...
    'PSO', 'GA', 'GWO', 'SSA', 'Location', 'best');  

%% 9. Print
fprintf('=========================================\n');
fprintf('Performance Comparison \n');  
fprintf('=========================================\n');

pso_len_raw = calculatePathLength(pso_fullPath);
pso_len_smooth = calculatePathLength(pso_smoothPath);
pso_hard_collision = isPathCollision(pso_smoothPath, obstacles, xGrid, yGrid, zTerrain);
pso_soft_collision = isPathSoftCollision(pso_smoothPath, softObstacles);

ga_len_raw = calculatePathLength(ga_fullPath);
ga_len_smooth = calculatePathLength(ga_smoothPath);
ga_hard_collision = isPathCollision(ga_smoothPath, obstacles, xGrid, yGrid, zTerrain);
ga_soft_collision = isPathSoftCollision(ga_smoothPath, softObstacles);

gwo_len_raw = calculatePathLength(gwo_fullPath);
gwo_len_smooth = calculatePathLength(gwo_smoothPath);
gwo_hard_collision = isPathCollision(gwo_smoothPath, obstacles, xGrid, yGrid, zTerrain);
gwo_soft_collision = isPathSoftCollision(gwo_smoothPath, softObstacles);

ssa_len_raw = calculatePathLength(ssa_fullPath);
ssa_len_smooth = calculatePathLength(ssa_smoothPath);
ssa_hard_collision = isPathCollision(ssa_smoothPath, obstacles, xGrid, yGrid, zTerrain);
ssa_soft_collision = isPathSoftCollision(ssa_smoothPath, softObstacles);

fprintf('| Algorithm              | PSO   | GA    | GWO   | SSA   |\n');  
fprintf('|--------------------|-------|-------|-------|-------|\n');
fprintf('| Fitness     | %.2f | %.2f | %.2f | %.2f |\n', ...
    pso_gbestFitness, ga_globalBestFitness, gwo_globalBestFitness, ssa_globalBestFitness);
fprintf('| Path Length(m)  | %.2f | %.2f | %.2f | %.2f |\n', ...
    pso_len_smooth, ga_len_smooth, gwo_len_smooth, ssa_len_smooth);
fprintf('| Hard Object Collisiion       | %s   | %s   | %s   | %s   |\n', ...
    bool2str(pso_hard_collision), bool2str(ga_hard_collision), bool2str(gwo_hard_collision), bool2str(ssa_hard_collision));
fprintf('| Soft Object Collisiion        | %s   | %s   | %s   | %s   |\n', ...
    bool2str(pso_soft_collision), bool2str(ga_soft_collision), bool2str(gwo_soft_collision), bool2str(ssa_soft_collision));
fprintf('=========================================\n');

%%  Function
function [pop, vel, pbest, pbestFitness, gbest, gbestFitness] = initPSO(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain, velMin, velMax)
    pop = zeros(popSize, pathNum, dim);
    vel = zeros(popSize, pathNum, dim);
    for i = 1:popSize
        for j = 1:pathNum
            pop(i,j,1) = startPoint(1) + (endPoint(1)-startPoint(1))*(j/pathNum) + randn()*1.5;
            pop(i,j,2) = startPoint(2) + (endPoint(2)-startPoint(2))*(j/pathNum) + randn()*1.5;
            zTerrainPoint = interp2(xGrid, yGrid, zTerrain, pop(i,j,1), pop(i,j,2), 'linear', 0);
            pop(i,j,3) = max(zTerrainPoint + 1, startPoint(3) + (endPoint(3)-startPoint(3))*(j/pathNum) + randn()*0.8);
        end
        pop(i,:,:) = smoothPathNodes(pop(i,:,:), 2);
        pop(i,:,1) = max(min(pop(i,:,1), ub(1)), lb(1));
        pop(i,:,2) = max(min(pop(i,:,2), ub(2)), lb(2));
        pop(i,:,3) = max(min(pop(i,:,3), ub(3)), lb(3));
    end
    for i = 1:popSize
        for j = 1:pathNum
            vel(i,j,:) = velMin + rand(1,dim).*(velMax - velMin);
        end
    end
    pbest = pop;
    pbestFitness = ones(popSize, 1) * Inf;
    gbest = zeros(1, pathNum, dim);
    gbestFitness = Inf;
end

function drawEnvironment(xRange, yRange, obstacles, softObstacles, startPoint, endPoint, color_seabed, color_hard_obs, color_soft_obs, edge_soft_obs, xGrid, yGrid, zTerrain);
    surf(xGrid, yGrid, zTerrain, 'FaceColor', color_seabed, 'FaceAlpha', 0.8, 'EdgeColor', 'none'); 
    
    for o = 1:size(obstacles,1)
        [xObs, yObs, zObs] = sphere(20);
        xObs = xObs*obstacles(o,4) + obstacles(o,1);
        yObs = yObs*obstacles(o,4) + obstacles(o,2);
        zObs = zObs*obstacles(o,4) + obstacles(o,3);
        surf(xObs, yObs, zObs, 'FaceColor', color_hard_obs, 'FaceAlpha', 0.6, 'EdgeColor', 'k');
    end
    
    for so = 1:size(softObstacles,1)
        [xSObs, ySObs, zSObs] = sphere(20);
        xSObs = xSObs*softObstacles(so,4) + softObstacles(so,1);
        ySObs = ySObs*softObstacles(so,4) + softObstacles(so,2);
        zSObs = zSObs*softObstacles(so,4) + softObstacles(so,3);
        surf(xSObs, ySObs, zSObs, 'FaceColor', color_soft_obs, 'FaceAlpha', 0.4, 'EdgeColor', edge_soft_obs);
    end
    
    scatter3(startPoint(1), startPoint(2), startPoint(3), 100, 'g', 'filled', 'MarkerEdgeColor', 'k');
    scatter3(endPoint(1), endPoint(2), endPoint(3), 100, 'b', 'filled', 'MarkerEdgeColor', 'k');
    text(startPoint(1), startPoint(2), startPoint(3)+3, 'Start', 'FontSize', 12);
    text(endPoint(1), endPoint(2), endPoint(3)+3, 'End', 'FontSize', 12);
end

function fit = calculateFitnessAUV(path, startPoint, endPoint, obstacles, softObstacles, xGrid, yGrid, zTerrain)
    lambda1 = 0.2;
    lambda2 = 0.1;
    lambda3 = 0.3;
    hardCollisionPenalty = 1000;
    softCollisionPenalty = 500;
    path_2d = reshape(path, [], 3);
    fullPath = [startPoint; path_2d; endPoint];
    nPoints = size(fullPath, 1);
    F_L = calculatePathLength(fullPath);
    F_D = 0;
    safeDepthToTerrain = 1.0;
    for i = 1:nPoints
        x = fullPath(i, 1);
        y = fullPath(i, 2);
        z = fullPath(i, 3);
        zTerrainPoint = interp2(xGrid, yGrid, zTerrain, x, y, 'linear', 0);
        distToTerrain = z - zTerrainPoint;
        if distToTerrain < safeDepthToTerrain
            F_D = F_D + (safeDepthToTerrain - distToTerrain)^2 * 20;
        else
            F_D = F_D + distToTerrain * 0.1;
        end
    end
    F_O = 0;
    safeDistToSoftObs = 2.0;
    for so = 1:size(softObstacles, 1)
        obsCenter = softObstacles(so, 1:3);
        obsRadius = softObstacles(so, 4);
        threatRange = obsRadius + safeDistToSoftObs;
        for i = 1:nPoints
            point = fullPath(i, :);
            distToSoftObs = norm(point - obsCenter);
            if distToSoftObs < threatRange
                F_O = F_O + (threatRange - distToSoftObs)^2 * 15;
            end
        end
    end
    collisionPenalty = 0;
    if isPathCollision(fullPath, obstacles, xGrid, yGrid, zTerrain)
        collisionPenalty = collisionPenalty + hardCollisionPenalty;
    end
    if isPathSoftCollision(fullPath, softObstacles)
        collisionPenalty = collisionPenalty + softCollisionPenalty;
    end
    for i = 1:nPoints
        zTerrainPoint = interp2(xGrid, yGrid, zTerrain, fullPath(i,1), fullPath(i,2), 'linear', 0);
        if fullPath(i,3) < zTerrainPoint + 0.5
            collisionPenalty = collisionPenalty + 500;
        end
    end
    fit = lambda1 * F_L + lambda2 * F_D + lambda3 * F_O + collisionPenalty;
end

function smoothNodes = smoothPathNodes(nodes, windowSize)
    [n, m, d] = size(nodes);
    smoothNodes = nodes;
    for k = 1:d
        smoothNodes(:,:,k) = movmean(squeeze(nodes(:,:,k)), windowSize, 2, 'Endpoints', 'shrink');
    end
end

function smoothPath = bsplineSmooth(path, numPoints)
    t = linspace(0, 1, size(path,1));
    tSmooth = linspace(0, 1, size(path,1)*numPoints);
    smoothPath = zeros(length(tSmooth), 3);
    for i = 1:3
        spl = spline(t, path(:,i));
        smoothPath(:,i) = ppval(spl, tSmooth);
    end
end

function len = calculatePathLength(fullPath)
    len = 0;
    for i = 2:size(fullPath,1)
        len = len + norm(fullPath(i,:) - fullPath(i-1,:));
    end
end

function isCol = isPathCollision(fullPath, obstacles, xGrid, yGrid, zTerrain)
    isCol = false;
    for o = 1:size(obstacles,1)
        obsCenter = obstacles(o,1:3);
        obsRadius = obstacles(o,4);
        for i = 2:size(fullPath,1)
            p1 = fullPath(i-1,:);
            p2 = fullPath(i,:);
            dist = pointToLineDistance(obsCenter, p1, p2);
            if dist < obsRadius + 0.5
                isCol = true;
                return;
            end
        end
    end
end

function isSoftCol = isPathSoftCollision(fullPath, softObstacles)
    isSoftCol = false;
    for so = 1:size(softObstacles,1)
        obsCenter = softObstacles(so,1:3);
        obsRadius = softObstacles(so,4);
        for i = 2:size(fullPath,1)
            p1 = fullPath(i-1,:);
            p2 = fullPath(i,:);
            dist = pointToLineDistance(obsCenter, p1, p2);
            if dist < obsRadius
                isSoftCol = true;
                return;
            end
        end
    end
end

function isThreat = isPathSoftThreat(fullPath, softObstacles)
    isThreat = false;
    safeDistToSoftObs = 2.0;
    for so = 1:size(softObstacles,1)
        obsCenter = softObstacles(so,1:3);
        obsRadius = softObstacles(so,4);
        threatRange = obsRadius + safeDistToSoftObs;
        for i = 1:size(fullPath,1)
            dist = norm(fullPath(i,:) - obsCenter);
            if dist < threatRange
                isThreat = true;
                return;
            end
        end
    end
end

function dist = pointToLineDistance(point, lineP1, lineP2)
    v = lineP2 - lineP1;
    w = point - lineP1;
    t = max(0, min(1, dot(w, v)/dot(v, v)));
    projPoint = lineP1 + t*v;
    dist = norm(point - projPoint);
end

function str = bool2str(bool)
    if bool
        str = 'YES';
    else
        str = 'NO';
    end
end

function [pop, globalBestFitness, globalBestPath] = initGA(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain)
    pop = zeros(popSize, pathNum, dim);
    for i = 1:popSize
        for j = 1:pathNum
            pop(i,j,1) = startPoint(1) + (endPoint(1)-startPoint(1))*(j/pathNum) + randn()*1.5;
            pop(i,j,2) = startPoint(2) + (endPoint(2)-startPoint(2))*(j/pathNum) + randn()*1.5;
            zTerrainPoint = interp2(xGrid, yGrid, zTerrain, pop(i,j,1), pop(i,j,2), 'linear', 0);
            pop(i,j,3) = max(zTerrainPoint + 1, startPoint(3) + (endPoint(3)-startPoint(3))*(j/pathNum) + randn()*0.8);
        end
        pop(i,:,:) = smoothPathNodes(pop(i,:,:), 2);
        pop(i,:,1) = max(min(pop(i,:,1), ub(1)), lb(1));
        pop(i,:,2) = max(min(pop(i,:,2), ub(2)), lb(2));
        pop(i,:,3) = max(min(pop(i,:,3), ub(3)), lb(3));
    end
    globalBestFitness = Inf;
    globalBestPath = [];
end

function selected = tournamentSelection(pop, fit, tournamentSize)
    popSize = size(pop, 1);
    selected = pop;
    for i = 1:popSize
        candidates = randi(popSize, 1, tournamentSize);
        [~, bestIdx] = min(fit(candidates));
        selected(i,:,:) = pop(candidates(bestIdx),:,:);
    end
end

function crossed = crossover(pop, crossoverProb, lb, ub)
    popSize = size(pop, 1);
    pathNum = size(pop, 2);
    dim = size(pop, 3);
    crossed = pop;
    
    for i = 1:2:popSize-1
        if rand() < crossoverProb
            crossPoint = randi(pathNum-1);
            temp = crossed(i, crossPoint+1:end, :);
            crossed(i, crossPoint+1:end, :) = crossed(i+1, crossPoint+1:end, :);
            crossed(i+1, crossPoint+1:end, :) = temp;
            
            crossed(i,:,1) = max(min(crossed(i,:,1), ub(1)), lb(1));
            crossed(i,:,2) = max(min(crossed(i,:,2), ub(2)), lb(2));
            crossed(i,:,3) = max(min(crossed(i,:,3), ub(3)), lb(3));
            crossed(i+1,:,1) = max(min(crossed(i+1,:,1), ub(1)), lb(1));
            crossed(i+1,:,2) = max(min(crossed(i+1,:,2), ub(2)), lb(2));
            crossed(i+1,:,3) = max(min(crossed(i+1,:,3), ub(3)), lb(3));
        end
    end
end

function mutated = mutation(pop, mutationProb, lb, ub, pathNum, dim)
    popSize = size(pop, 1);
    mutated = pop;
    
    for i = 1:popSize
        if rand() < mutationProb
            mutateNode = randi(pathNum);
            disturb = reshape((ub - lb) .* 0.1 .* randn(1, dim), 1, 1, 3);
            mutated(i, mutateNode, :) = mutated(i, mutateNode, :) + disturb;
            
            mutated(i, mutateNode, 1) = max(min(mutated(i, mutateNode, 1), ub(1)), lb(1));
            mutated(i, mutateNode, 2) = max(min(mutated(i, mutateNode, 2), ub(2)), lb(2));
            mutated(i, mutateNode, 3) = max(min(mutated(i, mutateNode, 3), ub(3)), lb(3));
        end
    end
end

function [pop, globalBestFitness, globalBestPath] = initGWO(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain)
    pop = zeros(popSize, pathNum, dim);
    for i = 1:popSize
        for j = 1:pathNum
            pop(i,j,1) = startPoint(1) + (endPoint(1)-startPoint(1))*(j/pathNum) + randn()*1.5;
            pop(i,j,2) = startPoint(2) + (endPoint(2)-startPoint(2))*(j/pathNum) + randn()*1.5;
            zTerrainPoint = interp2(xGrid, yGrid, zTerrain, pop(i,j,1), pop(i,j,2), 'linear', 0);
            pop(i,j,3) = max(zTerrainPoint + 1, startPoint(3) + (endPoint(3)-startPoint(3))*(j/pathNum) + randn()*0.8);
        end
        pop(i,:,:) = smoothPathNodes(pop(i,:,:), 2);
        pop(i,:,1) = max(min(pop(i,:,1), ub(1)), lb(1));
        pop(i,:,2) = max(min(pop(i,:,2), ub(2)), lb(2));
        pop(i,:,3) = max(min(pop(i,:,3), ub(3)), lb(3));
    end
    globalBestFitness = Inf;
    globalBestPath = [];
end

function [pop, globalBestFitness, globalBestPath] = initSSA(popSize, pathNum, dim, lb, ub, startPoint, endPoint, xGrid, yGrid, zTerrain)
    pop = zeros(popSize, pathNum, dim);
    for i = 1:popSize
        for j = 1:pathNum
            pop(i,j,1) = startPoint(1) + (endPoint(1)-startPoint(1))*(j/pathNum) + randn()*1.5;
            pop(i,j,2) = startPoint(2) + (endPoint(2)-startPoint(2))*(j/pathNum) + randn()*1.5;
            zTerrainPoint = interp2(xGrid, yGrid, zTerrain, pop(i,j,1), pop(i,j,2), 'linear', 0);
            pop(i,j,3) = max(zTerrainPoint + 1, startPoint(3) + (endPoint(3)-startPoint(3))*(j/pathNum) + randn()*0.8);
        end
        pop(i,:,:) = smoothPathNodes(pop(i,:,:), 2);
        pop(i,:,1) = max(min(pop(i,:,1), ub(1)), lb(1));
        pop(i,:,2) = max(min(pop(i,:,2), ub(2)), lb(2));
        pop(i,:,3) = max(min(pop(i,:,3), ub(3)), lb(3));
    end
    globalBestFitness = Inf;
    globalBestPath = [];
end
