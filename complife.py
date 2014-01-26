import os
import sys
import numpy as np
import pygame
import hashlib

# display constants
cell_width = 10
cell_height = 10

def hash_array(a):
    return hashlib.sha1(a.view(np.uint8)).hexdigest()

def grid_score(grid_cell,grid_color):
    on_mask = (grid_cell == 1)
    if on_mask.any():
        return np.mean(grid_color[grid_cell==1])
    else:
        return 0.5

def simulate_grid(grid_cell,grid_color,max_rep=2000,max_cycle=100,interact=False,runit=True,tock=50,paused=False):
    grid_shape = grid_cell.shape
    (grid_height,grid_width) = grid_shape

    if interact:
        BLK_COLOR = (0,0,0)
        SCREEN_WIDTH = cell_width*grid_width
        SCREEN_HEIGHT = cell_height*grid_height

        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT),0,32)
        clock = pygame.time.Clock()

    hash_list = np.zeros(max_cycle,dtype='|S80')
    hash_pos = 0

    do_exit = False
    found_cycle = False
    cycles_left = None

    for rep in range(max_rep):
        if interact:
            if paused:
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_SPACE:
                            paused = False
                            break
                        if event.key == pygame.K_TAB:
                            break
                        if event.key == pygame.K_q:
                            do_exit = True
                            break

            # Limit frame speed to 50 FPS
            time_passed = clock.tick(tock)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.KEYUP:
                    print (event.key,pygame.K_q)
                    if event.key == pygame.K_SPACE:
                        paused = True
                    if event.key == pygame.K_q:
                        do_exit = True

            if do_exit:
                print 'Quitting'
                break

            # Redraw the background
            screen.fill(BLK_COLOR)

            for (i,row) in enumerate(grid_cell):
                for (j,v) in enumerate(row):
                    if v == 1:
                        r = pygame.Rect(j*cell_width,i*cell_height,cell_width,cell_height)
                        ci = grid_color[i,j]
                        c = (int(255*(1.0-ci)),0,int(255*ci))
                        screen.fill(c,rect=r)

            pygame.display.flip()

        if not runit:
            break

        # count up neighbors
        nbr_counts = np.zeros(grid_shape,dtype=np.int)
        nbr_counts[1:,1:] += grid_cell[:-1,:-1] # top-left
        nbr_counts[1:,:] += grid_cell[:-1,:] # top-middle
        nbr_counts[1:,:-1] += grid_cell[:-1,1:] # top-right
        nbr_counts[:,1:] += grid_cell[:,:-1] # middle-left
        nbr_counts[:,:-1] += grid_cell[:,1:] # middle-right
        nbr_counts[:-1,1:] += grid_cell[1:,:-1] # bottom-left
        nbr_counts[:-1,:] += grid_cell[1:,:] # bottom-middle
        nbr_counts[:-1,:-1] += grid_cell[1:,1:] # bottom-right

        nbr_color = np.zeros(grid_shape)
        nbr_color[1:,1:] += grid_color[:-1,:-1] # top-left
        nbr_color[1:,:] += grid_color[:-1,:] # top-middle
        nbr_color[1:,:-1] += grid_color[:-1,1:] # top-right
        nbr_color[:,1:] += grid_color[:,:-1] # middle-left
        nbr_color[:,:-1] += grid_color[:,1:] # middle-right
        nbr_color[:-1,1:] += grid_color[1:,:-1] # bottom-left
        nbr_color[:-1,:] += grid_color[1:,:] # bottom-middle
        nbr_color[:-1,:-1] += grid_color[1:,1:] # bottom-right
        nbr_color /= nbr_counts

        # implement cell changes
        birth = (nbr_counts==3) & (grid_cell==0)
        death = ((nbr_counts>=4)|(nbr_counts<=1)) & (grid_cell==1)

        grid_cell[birth] = 1
        grid_cell[death] = 0
        grid_color[birth] = nbr_color[birth]
        grid_color[death] = 0.0

        # if (rep%100)==0:
        #     print '%i: %f' % (rep,np.mean(grid_color[grid_cell==1]))

        if found_cycle:
            mean_track += grid_score(grid_cell,grid_color)/found_cycle
            cycles_left -= 1
            if cycles_left == 0: break
        else:
            hash_val = hash_array(grid_cell) + hash_array(grid_color)
            if (hash_val==hash_list).any():
                found_cycle = (hash_val==np.concatenate((hash_list[hash_pos-1::-1],hash_list[-1:hash_pos-1:-1]))).nonzero()[0][0]+1
                #print 'Cycle of length %i' % found_cycle
                cycles_left = found_cycle
                mean_track = 0.0
            else:
                hash_list[hash_pos] = hash_val
                hash_pos = (hash_pos+1) % max_cycle

    if not found_cycle:
        mean_track = grid_score(grid_cell,grid_color)

    if interact:
        while True:
            event = pygame.event.wait()
            if event.type == pygame.KEYUP:
                break
        pygame.quit()

    #print 'Exiting'
    return mean_track

def random_player(grid_shape,density=0.5):
    return (np.random.random(grid_shape)<density).astype(np.int)

def load_player(fname):
    return np.loadtxt('save/'+fname,dtype=np.int)

def save_player(fname,p):
    np.savetxt('save/'+fname,p,fmt='%1d')

def load_panel(pname):
    fnames = os.listdir('save/'+pname)
    return [load_player(pname+'/'+fn) for fn in fnames]

def save_panel(players,pname):
    if not os.path.exists('save/'+pname): os.mkdir('save/'+pname)
    for (i,p) in enumerate(players):
        save_player(pname+'/'+('player_%d.lif'%i),p)

def run_match(player_left,player_right,**kwargs):
    player_flip = player_right[:,-1::-1]
    grid_cell = np.hstack((player_left,player_flip))
    grid_color = np.hstack((1.0*player_left,0.0*player_flip))
    return simulate_grid(grid_cell,grid_color,**kwargs)

def mate(player_one,player_two):
    grid_shape = player_one.shape
    grid_size = grid_shape[0]

    player_new = np.zeros(grid_shape)

    #inter = np.random.rand()
    #slope = 2.0*(np.random.rand()-0.5)
    #indices = [(i,j) for i in np.linspace(0.0,1.0,grid_size) for j in np.linspace(0.0,1.0,grid_size)]
    #sel_one = zip(*filter(lambda v: v[0]>=inter+slope*v[1],indices))
    #sel_two = zip(*filter(lambda v: v[0]<inter+slope*v[1],indices))
    #player_new[sel_one] = player_one[sel_one]
    #player_new[sel_two] = player_one[sel_two]

    cutline = np.random.randint(grid_size)
    player_new[:cutline] = player_one[:cutline]
    player_new[cutline:] = player_two[cutline:]

    return player_new

def mutate(player,factor):
    mutfact = 2.0*(np.random.rand()-0.5)*factor
    if mutfact > 0.0:
        cells_off = (1-player).nonzero()
        player[cells_off] = (np.random.rand(len(cells_off[0]))<mutfact).astype(np.int)
    else:
        cells_on = player.nonzero()
        player[cells_on] = (np.random.rand(len(cells_on[0]))>=(-mutfact)).astype(np.int)
    return player

def make_slate(base_players,n_players):
    n_base = len(base_players)
    all_players = [load_player(bp) if type(bp) is str else bp for bp in base_players]
    grid_shape = all_players[0].shape
    all_players += [random_player(grid_shape) for i in range(n_players-n_base)]
    return all_players

def repopulate(base_players,n_players,mutate_factor=0.0):
    n_base = len(base_players)
    new_players = []
    for i in range(n_players-n_base):
        challenge = np.random.choice(range(n_base),size=2,replace=False)
        new_players.append(mutate(mate(*[base_players[i] for i in challenge]),mutate_factor))
    return base_players + new_players

def tourney(all_players,n_games=10):
    n_players = len(all_players)
    scores = np.zeros(n_players)
    for (i1,p1) in enumerate(all_players):
        opps = np.random.randint(n_players,size=n_games)
        scores[i1] = np.mean([run_match(p1,all_players[i2]) for i2 in opps])
    return scores

def evolve(in_players,n_players,n_games=10,n_elim=0.8,mutate_factor=0.2):
    if type(n_elim) is float: n_elim = n_elim*float(n_players)
    all_players = repopulate(in_players,n_players,mutate_factor=mutate_factor)
    scores = tourney(all_players,n_games)
    survive = np.argsort(scores)[n_elim:]
    return ([all_players[i] for i in survive],[scores[i] for i in survive])

def iterate(in_panel,out_panel,max_rep,n_players=100,selection=0.25,mutation=0.4):
    zero_players = load_panel(in_panel)
    n_players = max(n_players,len(zero_players))

    mean_scores = np.zeros(max_rep)
    for rep in range(max_rep):
        (new_players,scores) = evolve(zero_players,n_players,n_elim=selection,mutate_factor=mutation)
        mscore = np.mean(scores)
        mdens = np.mean([np.mean(p) for p in new_players])

        print '%4i: %f, %f' % (rep,mscore,mdens)

        zero_players = new_players
        mean_scores[rep] = mscore

    save_panel(new_players,out_panel)
