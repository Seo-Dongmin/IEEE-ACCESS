function [gradients,loss] = modelGradients(NN_Net,dlX,Y)

    dlYPred = forward(NN_Net,dlX);
    dlYPred = sigmoid(dlYPred);
    
    loss = crossentropy(dlYPred,Y, 'TargetCategories','independent');
    
    gradients = dlgradient(loss,NN_Net.Learnables);

end
