function [gradients,loss] = modelGradients(MIMOME_Net,dlX,Y)

    dlYPred = forward(MIMOME_Net,dlX);
    dlYPred = sigmoid(dlYPred);
    
    loss = crossentropy(dlYPred,Y, 'TargetCategories','independent');
    
    gradients = dlgradient(loss,MIMOME_Net.Learnables);

end
