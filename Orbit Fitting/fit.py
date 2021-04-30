import orbitize
from orbitize import read_input, system, priors, sampler,results
import h5py
import os


def main():
	#parameters for the system 
	num_planets=1
	data_table = read_input.read_file('data.csv')
	m0 = 1.01
	mass_err = 0.05
	plx=78.33591471044681
	plx_err=0.1

	#initialise a system object
	sys = system.System(
	    num_planets, data_table, m0,
	    plx, mass_err=mass_err, plx_err=plx_err,fit_secondary_mass=True
	)
	
	sys.sys_priors[lab['plx1']] = priors.UniformPrior(60, 110)
	sys.sys_priors[lab['sma1']] = priors.UniformPrior(0.5,1.50)



	#MCMC parameters
	n_temps=10
	n_walkers=1000
	n_threads=10
	total_orbits_MCMC=75000000
	burn_steps=15000
	thin=10
	#set up sampler object and run it 
	mcmc_sampler = sampler.MCMC(sys,n_temps,n_walkers,n_threads)
	orbits=mcmc_sampler.run_sampler(total_orbits_MCMC,burn_steps=burn_steps,thin=thin)
	#save results
	myResults=mcmc_sampler.results
	try:
		### CHANGE THIS TO SAVE TO YOUR DESIRED DIRECTORY ##
		save_path = '/data/user/vnagpal/new_104304'
		filename  = 'floatplx.hdf5'
		hdf5_filename=os.path.join(save_path,filename)
		myResults.save_results(hdf5_filename)  # saves results object as an hdf5 file
	except:
		print("Something went wrong while saving the results")
	finally:      
		corner_figure=myResults.plot_corner()
		corner_name='floatplx_corner.png'
		corner_figure.savefig(corner_name) 
		orbit_figure=myResults.plot_orbits(rv_time_series=True)
		orbit_name='floatplx_orbit.png'
		orbit_figure.savefig(orbit_name)  

	return None

if __name__ == '__main__':
	main()


