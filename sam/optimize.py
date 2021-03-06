import numpy as np
import sam.log as log


def ravel(x):
    """ Convert value x to a flattened form (linear list). """
    return np.asarray(x).ravel()


class ModelParameterAcessor(object):
    def __init__(self, model, param_name):
        self.model = model
        self.param_name = param_name

        param = getattr(model, param_name)
        self.is_scalar = np.isscalar(param)
        self.param_shape = param.shape if not self.is_scalar else None

    def get(self):
        return getattr(self.model, self.param_name)

    def set(self, value):
        setattr(self.model, self.param_name, value)

    def get_flattened(self):
        """ Gets a flattened, linear representation of the parameter value. """
        return ravel(self.get())

    def set_flattened(self, value):
        """ Sets the parameter to the flattened value by reshaping it to the parameter's shape. """
        return self.set(self.unravel(value))

    def unravel(self, x):
        """ Reshape a flat representation of the parameter value into the parameter's expected shape. """
        x = np.asarray(x)
        if self.is_scalar:
            return x.item()
        else:
            return np.array(x).reshape(self.param_shape)


def check_grad(model, param_name, f, g):
    from scipy.optimize import check_grad as scipy_check_grad

    p = ModelParameterAcessor(model, param_name)

    def eval_f(param_as_list):
        old_value = p.get()  # Save old
        p.set_flattened(param_as_list)  # Set new
        f_val = f()
        p.set(old_value)  # Restore old value
        return f_val

    def eval_g(param_as_list):
        old_value = p.get()  # Save old
        p.set_flattened(param_as_list)  # Set new
        g_val = ravel(g())
        p.set(old_value)  # Restore old value
        return g_val

    x0 = ravel(p.get())
    return scipy_check_grad(eval_f, eval_g, x0)


def optimize_parameter(model, param_name, f, g, bounds=(1e-4, None), disp=0, max_evals=100):
    from scipy.optimize import fmin_tnc

    p = ModelParameterAcessor(model, param_name)

    # Scipy expects function parameters to be 1d, so we have to ravel/unravel the parameter values for each
    # evaluation
    def negative_f_and_f_prime(param_as_list):
        old_value = p.get()  # Save old
        p.set_flattened(param_as_list)  # Set new
        f_val = -f()
        f_prime_val = ravel(-g())
        p.set(old_value)  # Restore old value
        return f_val, f_prime_val

    x0 = ravel(p.get())
    bounds = [bounds] * len(x0)

    old_f_val = f()
    x, nfeval, rc = fmin_tnc(negative_f_and_f_prime, x0=x0, bounds=bounds, disp=disp, maxfun=max_evals)
    p.set_flattened(x)
    new_f_val = f()
    log.info('Optimized %s; improvement: %g' % (param_name, new_f_val - old_f_val))

def optimize_parameter_lbfgs_coor(model, f, g, bounds=(1e-4, None), disp=0, max_evals=10):
    from scipy.optimize import fmin_l_bfgs_b

    doc_x = ModelParameterAcessor(model, 'x')
    topic_delta = ModelParameterAcessor(model, 'delta')

    # Scipy expects function parameters to be 1d, so we have to ravel/unravel the parameter values for each
    # evaluation
    def eval_f(param_as_list):
        old_value_doc_x = doc_x.get()  # Save old
        old_value_topic_delta = topic_delta.get()
        doc_x.set_flattened(param_as_list[:model.num_docs * model.dim])  # Set new
        topic_delta.set_flattened(param_as_list[model.num_docs * model.dim:])
        f_val = -f()
        doc_x.set(old_value_doc_x)  # Restore old value
        topic_delta.set(old_value_topic_delta)
        return f_val

    def eval_g(param_as_list):
        old_value_doc_x = doc_x.get()  # Save old
        old_value_topic_delta = topic_delta.get()
        doc_x.set_flattened(param_as_list[:model.num_docs * model.dim])  # Set new
        topic_delta.set_flattened(param_as_list[model.num_docs * model.dim:])
        f_prime_val = -g()
        doc_x.set(old_value_doc_x)  # Restore old value
        topic_delta.set(old_value_topic_delta)
        return f_prime_val

    x0 = np.concatenate([ravel(doc_x.get()),ravel(topic_delta.get())])
#     bounds = [bounds] * len(x0)

    old_f_val = f()
    x, new_f_val, d = fmin_l_bfgs_b(eval_f, x0, fprime=eval_g, maxfun=max_evals, disp=disp)
    doc_x.set_flattened(x[:model.num_docs * model.dim])
    topic_delta.set_flattened(x[model.num_docs * model.dim:])
    new_f_val = f()
    log.info('Optimized %s; improvement: %g' % ('x,delta', new_f_val - old_f_val))
	
def optimize_parameter_lbfgs(model, param_name, f, g, bounds=(1e-4, None), disp=0, max_evals=100):
    from scipy.optimize import fmin_l_bfgs_b

    p = ModelParameterAcessor(model, param_name)

    # Scipy expects function parameters to be 1d, so we have to ravel/unravel the parameter values for each
    # evaluation
    def eval_f(param_as_list):
        old_value = p.get()  # Save old
        p.set_flattened(param_as_list)  # Set new
        f_val = f()
        p.set(old_value)  # Restore old value
        return -f_val

    def eval_g(param_as_list):
        old_value = p.get()  # Save old
        p.set_flattened(param_as_list)  # Set new
        g_val = ravel(g())
        p.set(old_value)  # Restore old value
        return -g_val

    x0 = ravel(p.get())
    #bounds = [(1e-4, None) for each in x0]  # Keep the parameter positive
    bounds = [bounds] * len(x0)

    old_f_val = f()
    x, new_f_val, d = fmin_l_bfgs_b(eval_f, x0, fprime=eval_g, bounds=bounds, maxfun=max_evals, disp=disp)
    p.set_flattened(x)
    new_f_val = f()
    log.info('Optimized %s; improvement: %g' % (param_name, new_f_val - old_f_val))
