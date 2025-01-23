# Reconhecimento de Imagens de Gatos ou Cães

Este é um programa para o reconhecimento de imagens de gatos ou cães utilizando redes neurais em Google Colab. Ele permite treinar um modelo de aprendizado de máquina para classificar imagens entre duas categorias: gato ou cão.
Estrutura do Projeto

    Subir as Imagens: As imagens precisam ser organizadas conforme a estrutura de pastas abaixo.

    /pets
        /train
            /cats
                cat1.jpg
                cat2.jpg
                ...
            /dogs
                dog1.jpg
                dog2.jpg
                ...
        /validation
            /cats
                cat101.jpg
                cat102.jpg
                ...
            /dogs
                dog101.jpg
                dog102.jpg
                ...

        A pasta train contém as imagens para treinamento do modelo.
        A pasta validation contém imagens para validação do modelo durante o treinamento.
        Dentro de cada pasta (cats e dogs), as imagens devem ser nomeadas de acordo com a convenção de sua escolha.

    Rodar o Treinamento: Após subir as imagens, execute o código de treinamento para gerar o modelo de aprendizado.
        O código irá ler as imagens das pastas train e validation, realizar o pré-processamento e treinar um modelo de rede neural para classificar as imagens como gato ou cão.
        O modelo será salvo em um arquivo com extensão .h5 (exemplo: modelo_cats_dogs.h5), que pode ser usado para futuras previsões.

    Rodar o Teste: Após o treinamento, você pode testar o modelo com novas imagens para verificar se ele classifica corretamente entre gatos e cães.
        A imagem deve ser fornecida em formato JPG ou PNG.
        O código de teste irá carregar a imagem, redimensioná-la e normalizá-la para que o modelo possa realizar a previsão.

Instruções de Uso
1. Subir as Imagens

    Crie a estrutura de pastas conforme mencionada acima e faça o upload das imagens para o Google Colab.
    Utilize a ferramenta de upload do Colab ou o módulo google.colab para organizar e carregar as imagens diretamente na máquina virtual do Colab.

2. Treinar o Modelo

Execute o código de treinamento no Colab para treinar o modelo com suas imagens.

    Código de treinamento: treinar.py



Após a execução, o modelo será salvo como um arquivo .h5 e estará pronto para ser utilizado.
3. Testar o Modelo

Para testar uma imagem, execute o código de teste abaixo, passando o caminho da imagem a ser testada.

    Código de teste: testar.py



A previsão será exibida no console, informando se a imagem contém um cachorro ou um gato.
Requisitos

    Python 3.x
    TensorFlow (Instalado automaticamente no Google Colab)
    Keras (Instalado automaticamente no Google Colab)
    Google Colab (Ambiente recomendado para rodar o código)

Considerações Finais

    Certifique-se de que as imagens estejam bem organizadas nas pastas corretas para garantir que o modelo seja treinado corretamente.
    O modelo pode ser melhorado com mais dados e ajustes nos hiperparâmetros do treinamento.

Se você tiver dúvidas ou problemas, sinta-se à vontade para abrir uma issue ou buscar ajuda na comunidade.