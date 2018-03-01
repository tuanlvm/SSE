import numpy as np
from scipy.special import gammaln, psi, polygamma
import sys

from sam.pickle_file_io import PickleFileIO
from sam.math_util import *
import sam.optimize as optimize
import sam.log as log
from sam.optimize import ravel


class SSE(PickleFileIO):
    def __init__(self, reader=None, T=None, Dim=None):
        assert reader is not None
        self.T = T if T is not None else 10 # Number of topics
        self.dim = Dim if Dim is not None else 2 # Dimension
        self.iteration = 0  # Number of iterations completed
        self.reader = reader
        self.corpus_file = self.reader.filename
        self.V = self.reader.dim  # Vocab size
        self.D = self.reader.num_docs
        self.num_docs = self.reader.num_docs

        self.load_corpus_as_matrix()

        # Variational parameters
        self.m = l2_normalize(np.ones(self.V))  # Parameter to p(mu)
        self.kappa0 = 10.0
        self.kappa1 = 5000.0
        self.xi = 5000.0

        self.vm = l2_normalize(np.random.rand(self.V))
        self.vmu = l2_normalize(np.random.rand(self.V, self.T))

        
        
        self.gamma = 0.1 * self.T
        self.beta = 0.1 * self.num_docs
        self.theta = np.empty((self.T, self.num_docs))
        self.theta_v = np.empty((self.T, self.num_docs))
        self.x = 0.01 * self.T * np.random.randn(self.dim,self.num_docs)
        self.delta = 0.01 * self.T * np.random.randn(self.dim,self.T)
        for d in range(self.num_docs):
            distances_from_topics = np.exp(-0.5 * np.sum((self.delta.T-self.x[:,d]).T ** 2,axis=0))
            self.theta[:,d] = distances_from_topics/np.sum(distances_from_topics)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.corpus_file = self.reader.filename
        self.load_corpus_as_matrix()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['v']
        return state

    def load_corpus_as_matrix(self):
        self.v = np.empty((self.V, self.num_docs))
        for d in xrange(self.num_docs):
            self.v[:, d] = self.reader.read_doc(d).T

    def l_vmu(self):
        a_xi = avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)

        sum_of_rhos = sum(self.rho_batch())

        vm_dot_sum_of_vmu = np.dot(self.vm.T, np.sum(self.vmu, axis=1))
        likelihood = a_xi*a_k0*self.xi*vm_dot_sum_of_vmu + self.kappa1*sum_of_rhos
        return likelihood
          
    def l_xi(self):
        a_xi = avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)
        sum_of_rhos = sum(self.rho_batch())
        print(a_xi*self.xi * (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T) \
            + self.kappa1*sum_of_rhos)
        return a_xi*self.xi * (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T) \
            + self.kappa1*sum_of_rhos

    def grad_l_vmu(self):
        a_xi = avk(self.V, self.xi)
        a_xi_squared = a_xi**2
        a_k0 = avk(self.V, self.kappa0)

        esns = self.e_squared_norm_batch()

        
        first_term = np.dot(self.v, (self.theta * a_xi / asrowvector(1 * np.sqrt(esns))).T)
        per_doc_weights = a_xi / (2*esns ** (3./2.)) \
                        * (self.theta * np.dot(self.vmu.T, self.v)).sum(axis=0).T

        third_term_doc_weights = per_doc_weights * 2*a_xi_squared
        third_term = np.dot(self.vmu, np.dot(self.theta * asrowvector(third_term_doc_weights), self.theta.T))

        sum_over_documents = first_term - third_term
        return ascolvector(a_xi*a_k0*self.xi*self.vm) + self.kappa1*sum_over_documents
    
    
    def grad_l_xi(self):
        a_xi = avk(self.V, self.xi)
        a_prime_xi = deriv_avk(self.V, self.xi)
        a_k0 = avk(self.V, self.kappa0)

        sum_over_documents = sum(self.deriv_rho_xi())
        return (a_prime_xi*self.xi + a_xi) * (a_k0*np.dot(self.vm.T, np.sum(self.vmu, axis=1)) - self.T) \
            + self.kappa1*sum_over_documents

    def tangent_grad_l_vmu(self):
        """
        The gradient of the likelihood bound with respect to vMu, projected into the tangent space of the hypersphere.
        """
        grad = self.grad_l_vmu()
        # Project the gradients into the tangent space at each topic
        for t in range(self.T):
            vmu_t = self.vmu[:, t]
            grad[:, t] = grad[:, t] - np.dot(vmu_t, np.dot(vmu_t.T, grad[:, t]))
        return grad
          
    def rho_batch(self):
        esns = self.e_squared_norm_batch()
        vmu_times_v = self.vmu.T.dot(self.v)
        return np.sum(self.theta * asrowvector(1.0/np.sqrt(esns)) * vmu_times_v, axis=0) \
                   * avk(self.V, self.xi)

    def e_squared_norm_batch(self):
        theta_squares = np.sum(self.theta**2, axis=0)
        a_xi_squared = avk(self.V, self.xi) ** 2

        vMuDotVMu = np.dot(self.vmu.T, self.vmu)  # T by T
        vMuThetaVMuTheta = np.sum(
            np.dot(self.theta.T, vMuDotVMu).T * self.theta,
            axis=0)

        esns = ((1.0-a_xi_squared)*theta_squares + a_xi_squared*vMuThetaVMuTheta)
        return esns

    def deriv_rho_xi(self):
        """ Gradient of each Rho_d with respect to xi. """
        a_xi = avk(self.V, self.xi)
        deriv_a_xi = deriv_avk(self.V, self.xi)
        esns = self.e_squared_norm_batch()
        deriv_e_squared_norm_xis  = self.grad_e_squared_norm_xi()
        
        vMuTimesThetaDotDoc = np.sum(self.theta * np.dot(self.vmu.T, self.v), axis=0)
        
        
        deriv = deriv_a_xi * vMuTimesThetaDotDoc / (1 * np.sqrt(esns)) \
            - a_xi/2 * vMuTimesThetaDotDoc / (1 * esns**1.5) * deriv_e_squared_norm_xis
        return deriv

    def grad_e_squared_norm_xi(self):
        """ Gradient of the expectation of the squared norms with respect to xi """
        a_xi = avk(self.V, self.xi)
        deriv_a_xi = deriv_avk(self.V, self.xi)
 
        sum_theta_squared = np.sum(self.theta**2, axis=0)
        vMuThetaVMuTheta = np.sum(np.dot(self.theta.T, np.dot(self.vmu.T, self.vmu)).T * self.theta, axis=0)
        deriv = 2*a_xi*deriv_a_xi*(vMuThetaVMuTheta - sum_theta_squared) 
        return deriv

    def update_vm(self):
        self.vm = l2_normalize(
            self.kappa0*self.m + avk(self.V, self.xi)*self.xi*np.sum(self.vmu, axis=1)
        )

    def update_m(self):
        self.m = l2_normalize(np.sum(self.vmu, axis=1))  # Sum across topics

    def update_xi(self):
        optimize.optimize_parameter(self, 'xi', self.l_xi, self.grad_l_xi)
        print (self.xi)

    def update_vmu(self):
        # XXX: The topics (vmus) must lie on the hypersphere, i.e. have unit L2 norm.  I'm not sure if scipy has
        # an optimization method that can accommodate this type of constraint, so instead, I'm encoding
        # it here a Lagrange multiplier.  This should at least push the optimizer towards solutions close to the
        # L2 constraint.

        # Set the strength of the Lagrange multipler to something much larger than the objective
        LAMBDA = 10.0*self.l_vmu()
        def f():
            squared_norms = np.sum(self.vmu ** 2, axis=0)
            return  self.l_vmu() - LAMBDA*np.sum((squared_norms - 1.0)**2)

        def g():
            squared_norms = np.sum(self.vmu ** 2, axis=0)
            return self.tangent_grad_l_vmu() - LAMBDA*2.0*(squared_norms - 1.0)*(2.0*self.vmu)

        optimize.optimize_parameter(self, 'vmu', f, g, bounds=(-1.0, 1.0))
        self.vmu = l2_normalize(self.vmu)  # Renormalize
    
    def l_coor(self):
        for d in range(self.num_docs):
            distances_from_topics = np.exp(-0.5 * np.sum((self.delta.T-self.x[:,d]).T ** 2,axis=0))
            self.theta[:,d] = distances_from_topics/np.sum(distances_from_topics)
                   
        sum_of_rhos = sum(self.rho_batch())
        likelihood = self.kappa1*sum_of_rhos + (-self.gamma/2) * np.sum(np.sum(self.x ** 2, axis=0)) + (-self.beta/2) * np.sum(np.sum(self.delta ** 2, axis=0))
        print(likelihood)
        return likelihood
    
    def grad_e_squared_norm_batch_x(self,gra_theta_x):
        a_xi_squared = avk(self.V, self.xi) ** 2
        grads = []
        for i in range(self.dim):
            grad = np.zeros(self.num_docs)
            for d in range(self.num_docs):
                vMuTimesThetaTimesVMu = np.dot(self.theta[:,d].T, np.dot(self.vmu.T, np.dot(self.vmu,gra_theta_x[i][:,d])))  # D by T
                per_doc_weights = 1./(1 * (1 + 1))
                gra = (2*(1-a_xi_squared)*np.sum(gra_theta_x[i][:,d]*self.theta[:,d],axis=0) + 2*a_xi_squared*vMuTimesThetaTimesVMu)
                grad[d] = gra
            grads.append(grad)
        return grads
      
    def grad_e_squared_norm_batch_delta(self, gra_theta_delta):
        a_xi_squared = avk(self.V, self.xi) ** 2
        grads = []
        for i in range(self.dim):
            grad = np.zeros((self.T,self.num_docs))
            for j in range(self.T):
                vMuTimesThetaTimesVMu = np.dot(self.theta.T, np.dot(self.vmu.T, np.dot(self.vmu,gra_theta_delta[i][j,:,:])))  # D by T
                gra = (2*(1-a_xi_squared)*np.sum(gra_theta_delta[i][j,:,:]*self.theta,axis=0) + 2*a_xi_squared*np.diagonal(vMuTimesThetaTimesVMu))
                grad[j] = gra
            grads.append(grad)
        return grads
    
    def grad_theta_x(self):
        grads=[]
        for d in range(self.dim):
            tile = np.tile(self.x[d,:], (self.T,1))
            x_minus_delta = tile - ascolvector(self.delta[d,:])
            first_term = self.theta * x_minus_delta
            grad = -first_term
            temp_sum = np.sum(first_term,axis=0)
            second_term = self.theta * temp_sum;
            grad = grad + second_term;
            grads.append(grad);
        return grads
    
    def grad_theta_delta(self):
        grads=[]
        for d in range(self.dim):
            grad = np.zeros((self.T,self.T,self.num_docs))
            for j in range(self.T):
                grad[j] = (-(self.x[d,:] - self.delta[d,j])*self.theta[j,:])*self.theta
                grad[j,j] = (self.x[d,:]-self.delta[d,j])*(self.theta[j,:] - self.theta[j,:]*self.theta[j,:])
            grads.append(grad)
        return grads
    
    def grad_l_x(self,esns,vMuDotVd,vMuTimesThetaDotVd):
        a_xi = avk(self.V, self.xi)
        gra_theta_x = self.grad_theta_x()
        derivsOfESquaredNorm = self.grad_e_squared_norm_batch_x(gra_theta_x);
        grads_of_rho = []
        s = vMuTimesThetaDotVd / (2*esns**(3./2.))
        for d in range(self.dim):
            grads_of_rho.append(self.kappa1 * a_xi * (np.sum(gra_theta_x[d] * vMuDotVd, axis=0)/np.sqrt(esns) - derivsOfESquaredNorm[d] * asrowvector(s)) - self.gamma * self.x[d,:])
        
        return np.concatenate(grads_of_rho[:])
     
    def grad_l_delta(self,esns,vMuDotVd,vMuTimesThetaDotVd):
        a_xi = avk(self.V, self.xi)
        gra_theta_delta = self.grad_theta_delta()
        derivsOfESquaredNorm = self.grad_e_squared_norm_batch_delta(gra_theta_delta)
        grads_of_rho=[]
        s = vMuTimesThetaDotVd / (2*esns**(3./2.))
        for d in range(self.dim):
            grads_of_rho.append(self.kappa1 * a_xi * (np.sum(gra_theta_delta[d] * vMuDotVd, axis=1)/asrowvector(np.sqrt(esns)) - derivsOfESquaredNorm[d] * asrowvector(s)))
        for d in range(self.dim):
            grads_of_rho[d] = np.sum(grads_of_rho[d],axis=1) - self.beta * self.delta[d,:]
            
        return np.concatenate([grads_of_rho[:]])
           
    def update_coordinates(self):
        def f():
            return self.l_coor()
        def g():
            esns = self.e_squared_norm_batch()
            vMuDotVd = np.dot(self.vmu.T, self.v)  # T by D
            vMuTimesThetaDotVd = np.sum(self.theta * vMuDotVd, axis=0)
            gx = self.grad_l_x(esns,vMuDotVd,vMuTimesThetaDotVd)
            gdelta = self.grad_l_delta(esns,vMuDotVd,vMuTimesThetaDotVd)
            return np.concatenate([ravel(gx),ravel(gdelta)])
        optimize.optimize_parameter_lbfgs_coor(self, f, g)
       
       
    def run_one_iteration(self):
        log.debug('Updating vMu')
        self.update_vmu()

        log.debug('Updating vM')
        self.update_vm()

        log.debug('Updating M')
        self.update_m()

        log.debug('Updating xi')
        self.update_xi()

        log.debug('Updating x,delta')
        self.update_coordinates()
        
        for d in range(self.num_docs):
            distances_from_topics = np.exp(-0.5 * np.sum((self.delta.T-self.x[:,d]).T ** 2,axis=0))
            self.theta[:,d] = distances_from_topics/np.sum(distances_from_topics)
        self.iteration += 1

    def write_topics(self, f=None, num_top_words=10, num_bottom_words=10):
        if f is None:
            f = sys.stdout

        wordlist = open(self.corpus_file + '.wordlist').readlines()
        wordlist = np.array([each.strip() for each in wordlist], str)

        for t in range(self.T):
            print >>f, 'Topic %d' % t
            print >>f, '--------'

            sorted_indices = np.argsort(self.vmu[:, t])
            sorted_weights = self.vmu[sorted_indices, t]
            sorted_words = wordlist[sorted_indices]

            print >>f, 'Top weighted words:'
            for word, weight in zip(sorted_words[:-num_top_words:-1], sorted_weights[:-num_top_words:-1]):
                print >>f, '  %.4f %s' % (weight, word)

            print >>f
            print >>f, 'Bottom weighted words:'
            for word, weight in zip(sorted_words[:num_bottom_words], sorted_weights[:num_bottom_words]):
                print >>f, '  %.4f %s' % (weight, word)

            print >>f
            print >>f

        if f is not sys.stdout:
            f.close()

    def write_topic_weights_arff(self, f=None):
        if f is None:
            f = sys.stdout

        mean_topic_weights = self.theta
        print >>f, 'vmu'
        for t in range(self.T):
            vmu_string = ', '.join([str(each) for each in self.vmu[:, t]])
            print >>f,vmu_string
            
        print >>f, 'vm'
        print >>f,', '.join([str(each) for each in self.vm])
        
        print >>f, 'm'
        print >>f,', '.join([str(each) for each in self.m])
        
        print >>f, 'xi'
        print >>f,self.xi
        
        print >>f, '@RELATION topicWeights'
        for t in range(self.T):
            print >>f, '@ATTRIBUTE topic%d NUMERIC' % t
        print >>f, '@ATTRIBUTE class {%s}' % ','.join(self.reader.class_names)
        print >>f, '@DATA'

        for d in range(self.num_docs):
            weights_string = ', '.join([str(each) for each in mean_topic_weights[:, d]])
            label = self.reader.raw_labels[d]
            name = self.reader.names[d]
            print >>f, '%s, %s, %s, %s, %s' % (name, weights_string, label,str(self.x[0,d]),str(self.x[1,d]))
        print >>f, 'topic_x\n'
        for t in range(self.T):
            print >>f, str(self.delta[0,t]) + '\t' + str(self.delta[1,t])
        if f is not sys.stdout:
            f.close()
